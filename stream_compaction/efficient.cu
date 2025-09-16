#include "common.cuh"
#include "efficient.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

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
         * @param g_arrayToScan the full array, in global memory. This will be modified in-place.
         *   This is expected to be of size N.
         * @param g_blockTotalSums where the full sums of each block will be saved.
         *   It is expected to be of size `N / blockDim.x`.
         */
        __global__ void kernExclusiveScanByBlocks(int N, int *g_arrayToScan,
                                                  int *g_blockTotalSums) {
            int blockStartIndex = (blockIdx.x * blockDim.x);
            int localThreadIndex = threadIdx.x;
            int globalThreadIndex = blockStartIndex + localThreadIndex;

            // TODO: is this log2 call ok?
            int maxDepth = log2(blockDim.x);

            // Do awesome upsweep in-place with increasing depth
            for (int d = 0; d < maxDepth; ++d) {
                int halfChunk = 1 << d;
                int fullChunk = halfChunk << 1;

                // Each layer gets blockSize >> (d + 1) operations
                int numThreads = blockDim.x / fullChunk;

                if (localThreadIndex < numThreads) {
                    // K is global index of first element of "chunk" we're operating on
                    int globalK = blockStartIndex + blockStartIndex + localThreadIndex * fullChunk;
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
                    int globalK = blockStartIndex + blockStartIndex + localThreadIndex * fullChunk;

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
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // int N = ilog2ceil(n) int

            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
        }

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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
