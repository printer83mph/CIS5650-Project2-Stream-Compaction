#include "common.cuh"
#include "efficient.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// this has gotta be a power of 2!
#define BLOCK_SIZE 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Perform segmented addition scan on an array, separated into blocks.
         *
         * @param N the number of total elements. Expected to be a power of 2.
         * @param maxDepth should be ilog2ceil(N). This should just be precomputed.
         * @param g_arrayToScan the full array, in global memory. This will be modified in-place.
         *   This is expected to be of size N.
         * @param g_blockTotalSums where the full sums of each block will be saved.
         *   It is expected to be of size `N / blockDim.x`.
         */
        __global__ void kernExclusiveScanByBlocks(int N, int maxDepth, int *g_arrayToScan,
                                                  int *g_blockTotalSums) {
            // TODO: probably use shared memory to make this faster, instead of pulling from evil
            // global world...

            int blockStartIndex = (blockIdx.x * blockDim.x);
            int localThreadIndex = threadIdx.x;
            int globalThreadIndex = blockStartIndex + localThreadIndex;

            // Do awesome upsweep in-place with increasing depth
            for (int d = 0; d < maxDepth; ++d) {
                int halfChunk = 1 << d;
                int fullChunk = halfChunk << 1;

                // Each layer gets blockSize >> (d + 1) operations
                int numThreads = blockDim.x / fullChunk;

                if (localThreadIndex < numThreads) {
                    // K is global index of first element of "chunk" we're operating on
                    int globalK = blockStartIndex + localThreadIndex * fullChunk;
                    g_arrayToScan[globalK + fullChunk - 1] +=
                        g_arrayToScan[globalK + halfChunk - 1];
                }
                __syncthreads();
            }

            // Save the last goober of each block for later, reset to 0 for down-sweep
            if (threadIdx.x == BLOCK_SIZE - 1) {
                g_blockTotalSums[blockIdx.x] = g_arrayToScan[globalThreadIndex];
                g_arrayToScan[globalThreadIndex] = 0;
            }
            __syncthreads();

            // Do awesome downsweep in-place with decreasing depth
            for (int d = maxDepth - 1; d >= 0; --d) {
                int halfChunk = 1 << d;
                int fullChunk = halfChunk << 1;

                // Each layer gets blockSize >> (d + 1) operations
                int numThreads = blockDim.x / fullChunk;

                if (localThreadIndex < numThreads) {
                    // K is global index of first element of "chunk" we're operating on
                    int globalK = blockStartIndex + localThreadIndex * fullChunk;

                    // Copy right value, add left one in-place, then set left to copied value
                    int oldRightValue = g_arrayToScan[globalK + fullChunk - 1];
                    g_arrayToScan[globalK + fullChunk - 1] +=
                        g_arrayToScan[globalK + halfChunk - 1];
                    g_arrayToScan[globalK + halfChunk - 1] = oldRightValue;
                }
                __syncthreads();
            }
        }

        /**
         * Perform exclusive scan on a single block of data.
         *
         * @param N the number of elements in the block. Expected to be a power of 2.
         * @param maxDepth should be ilog2ceil(N). This should just be precomputed.
         * @param g_data the array to scan in global memory. This will be modified in-place.
         */
        __global__ void kernExclusiveScanOneBlock(int N, int maxDepth, int *g_arrayToScan) {
            // TODO: probably use shared memory to make this faster, instead of pulling from evil
            // global world...

            int localThreadIndex = threadIdx.x;

            // Do upsweep in-place with increasing depth
            for (int d = 0; d < maxDepth; ++d) {
                int halfChunk = 1 << d;
                int fullChunk = halfChunk << 1;

                // Each layer gets blockSize >> (d + 1) operations
                int numThreads = blockDim.x / fullChunk;

                if (localThreadIndex < numThreads) {
                    // K is index of first element of "chunk" we're operating on
                    int k = localThreadIndex * fullChunk;
                    g_arrayToScan[k + fullChunk - 1] += g_arrayToScan[k + halfChunk - 1];
                }
                __syncthreads();
            }

            // Reset the last element to 0 for down-sweep
            if (threadIdx.x == 0) {
                g_arrayToScan[N - 1] = 0;
            }
            __syncthreads();

            // Do downsweep in-place with decreasing depth
            for (int d = maxDepth - 1; d >= 0; --d) {
                int halfChunk = 1 << d;
                int fullChunk = halfChunk << 1;

                // Each layer gets blockSize >> (d + 1) operations
                int numThreads = blockDim.x / fullChunk;

                if (localThreadIndex < numThreads) {
                    // K is index of first element of "chunk" we're operating on
                    int k = localThreadIndex * fullChunk;

                    // Copy right value, add left one in-place, then set left to copied value
                    int oldRightValue = g_arrayToScan[k + fullChunk - 1];
                    g_arrayToScan[k + fullChunk - 1] += g_arrayToScan[k + halfChunk - 1];
                    g_arrayToScan[k + halfChunk - 1] = oldRightValue;
                }
                __syncthreads();
            }
        }

        __global__ void kernAddChunkedSums(int N, int *g_chunkScannedArray, int *g_blockTotalSums) {
            int globalThreadIndex = threadIdx.x + (blockIdx.x * blockDim.x);
            g_chunkScannedArray[globalThreadIndex] += g_blockTotalSums[blockIdx.x];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool startGpuTimer) {
            int maxDepth = ilog2ceil(n);
            int N = 1 << maxDepth;

            // Pointers to all the extra scans needed when block size is smaller than N
            std::vector<int *> dev_arrays;

            // Create device mem pointers with decreasing sizes
            int currentSize = N;
            while (currentSize > BLOCK_SIZE) {
                int *dev_array;
                cudaMalloc(&dev_array, currentSize * sizeof(int));
                dev_arrays.push_back(dev_array);
                currentSize = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            }

            // Allocate one just for the smallest, always occurs
            int *dev_smallest_array;
            cudaMalloc(&dev_smallest_array, currentSize * sizeof(int));
            dev_arrays.push_back(dev_smallest_array);

            // Copy input data to first device array
            cudaMemcpy(dev_arrays[0], idata, n * sizeof(int), cudaMemcpyHostToDevice);
            // Fill rest of first array with zeros if n < N
            if (n < N) {
                cudaMemset(dev_arrays[0] + n, 0, (N - n) * sizeof(int));
            }

            if (startGpuTimer)
                timer().startGpuTimer();

            // Iterate through arrays from largest to smallest, performing scan on each level
            currentSize = N;
            for (int i = 0; i < dev_arrays.size() - 1; ++i) {
                int blocksPerGrid = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernExclusiveScanByBlocks<<<blocksPerGrid, BLOCK_SIZE>>>(
                    currentSize, ilog2(currentSize), dev_arrays[i], dev_arrays[i + 1]);
                checkCUDAError("kernExclusiveScanByBlocks failed!");
                cudaDeviceSynchronize();

                currentSize = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            }

            // Run scan on smallest block of all time
            kernExclusiveScanOneBlock<<<1, BLOCK_SIZE>>>(currentSize, ilog2(currentSize),
                                                         dev_smallest_array);
            checkCUDAError("kernExclusiveScanOneBlock failed!");
            cudaDeviceSynchronize();

            // Iterate back up through arrays from smallest to largest, adding chunk sums
            for (int i = dev_arrays.size() - 2; i >= 0; --i) {
                currentSize *= BLOCK_SIZE;

                int blocksPerGrid = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernAddChunkedSums<<<blocksPerGrid, BLOCK_SIZE>>>(currentSize, dev_arrays[i],
                                                                  dev_arrays[i + 1]);
                checkCUDAError("kernAddChunkedSums failed!");
                cudaDeviceSynchronize();
            }

            if (startGpuTimer)
                timer().endGpuTimer();

            // Copy result back to host
            cudaMemcpy(odata, dev_arrays[0], n * sizeof(int), cudaMemcpyDeviceToHost);

            // Deallocate all device pointers
            for (int *devicePtr : dev_arrays) {
                cudaFree(devicePtr);
            }
        }

        // by default: run GPU timer
        void scan(int n, int *odata, const int *idata) { scan(n, odata, idata, true); }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            int *dev_idata;
            int *dev_bools;
            int *dev_indices;
            int *dev_odata;
            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_bools, n * sizeof(int));
            cudaMalloc(&dev_indices, n * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));

            // Copy stuffs over to GPU world
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy from idata to dev_idata failed!");

            timer().startGpuTimer();

            // Map our input numbers to bools
            Common::kernMapToBoolean<<<blocksPerGrid, BLOCK_SIZE>>>(n, dev_bools, dev_idata);
            cudaDeviceSynchronize();
            checkCUDAError("kernMapToBoolean failed!");

            // Scan em into dev_indices!!!
            Efficient::scan(n, dev_indices, dev_bools, false);

            // Now scatter into the output
            Common::kernScatter<<<blocksPerGrid, BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_bools,
                                                               dev_indices);
            cudaDeviceSynchronize();
            checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from dev_odata to odata failed!");

            // Get the count of compacted elements
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, &dev_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            int matchingCount = lastIndex + lastBool;

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return matchingCount;
        }
    }
}
