#include <cuda.h>
#include <cuda_runtime.h>
#include "common.cuh"
#include "naive.cuh"

#define BLOCK_SIZE 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernShiftForExclusiveScan(int N, int *dest, int *src) {
            int threadIndex = threadIdx.x + (blockIdx.x * blockDim.x);
            if (threadIndex >= N)
                return;

            // Set up first element as zero, shift everything else right
            if (threadIndex == 0) {
                dest[0] = 0;
            } else if (threadIndex < N) {
                dest[threadIndex] = src[threadIndex - 1];
            }
        }

        __global__ void kernScanIteration(int N, int offset, int *dest, int *src) {
            int threadIndex = threadIdx.x + (blockIdx.x * blockDim.x);
            if (threadIndex >= N)
                return;

            if (threadIndex >= offset) {
                dest[threadIndex] = src[threadIndex] + src[threadIndex - offset];
            } else {
                dest[threadIndex] = src[threadIndex];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            dim3 blocksPerGrid = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int *dev_scanA;
            int *dev_scanB;
            cudaMalloc(&dev_scanA, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_scanA failed!");
            cudaMalloc(&dev_scanB, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_scanB failed!");

            // Send our input info over to the GPU
            cudaMemcpy(dev_scanA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            // Shift things over by one index
            kernShiftForExclusiveScan<<<blocksPerGrid, BLOCK_SIZE>>>(n, dev_scanB, dev_scanA);
            checkCUDAError("kernShiftForExclusiveScan failed!");

            int *src = dev_scanB;
            int *dest = dev_scanA;

            // Run scan iteratively
            for (int offset = 1; offset < n; offset *= 2) {
                kernScanIteration<<<blocksPerGrid, BLOCK_SIZE>>>(n, offset, dest, src);
                checkCUDAError("kernScanIteration failed!");

                // Swap src and dest pointers for next iteration
                int *temp = src;
                src = dest;
                dest = temp;
            }

            cudaDeviceSynchronize();

            // Take our output data back to the CPU
            cudaMemcpy(odata, src, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_scanA);
            cudaFree(dev_scanB);

            timer().endGpuTimer();
        }
    }
}
