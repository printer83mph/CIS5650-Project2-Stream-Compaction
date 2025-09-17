#include "common.cuh"
#include "efficient.cuh"
#include "radix.cuh"

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

namespace StreamCompaction {
namespace Radix {

using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
    static PerformanceTimer timer;
    return timer;
}

void cpu_sort(int n, int *odata, const int *idata) {
    int *idata_mut = new int[n];
    int *odata_mut = odata;

    // copy
    std::copy(idata, idata + n, idata_mut);

    Radix::timer().startCpuTimer();

    for (int bit = 0; bit < 32; ++bit) {
        // for each bit, ping-pong
        int currentIndex = 0;

        for (int i = 0; i < n; ++i) {
            int shouldPartitionRight = (idata_mut[i] >> bit) & 1;
            if (shouldPartitionRight)
                continue;

            odata_mut[currentIndex] = idata_mut[i];
            currentIndex++;
        }

        for (int i = 0; i < n; ++i) {
            int shouldPartitionRight = (idata_mut[i] >> bit) & 1;
            if (!shouldPartitionRight)
                continue;

            odata_mut[currentIndex] = idata_mut[i];
            currentIndex++;
        }

        int *temp = idata_mut;
        idata_mut = odata_mut;
        odata_mut = temp;
    }

    // Copy final result to odata if pinged to the wrong pong
    if (idata_mut != odata) {
        std::copy(idata_mut, idata_mut + n, odata);
    }

    Radix::timer().endCpuTimer();

    delete[] idata_mut;
}

/**
 *  Runs custom scatter operation for radix sort partitioning.
 *
 * @param falseIndices  An exclusive scan of the `bools` array
 * @param totalFalses   The total number of 1 bits in `bools`
 */
__global__ void kernCustomScatter(int n, int *odata, const int *idata, const int *falseBools,
                                  const int *falseIndices, int totalFalses) {
    int threadIndex = threadIdx.x + (blockIdx.x * blockDim.x);
    if (threadIndex >= n)
        return;

    int falseIndex = falseIndices[threadIndex];
    int trueIndex = threadIndex - falseIndex + totalFalses;

    int outputIndex = falseBools[threadIndex] ? falseIndex : trueIndex;
    odata[outputIndex] = idata[threadIndex];
}

/**
 * Performs radix sort on idata, storing the result into odata.
 */
void sort(int n, int *odata, const int *idata) {
    dim3 blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *dev_input;
    int *dev_output;
    int *dev_falseBools;
    int *dev_falseIndices;

    cudaMalloc(&dev_input, n * sizeof(int));
    cudaMalloc(&dev_output, n * sizeof(int));
    cudaMalloc(&dev_falseBools, n * sizeof(int));
    cudaMalloc(&dev_falseIndices, n * sizeof(int));

    // Copy input data to device
    cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

    timer().startGpuTimer();

    // Process each bit position (32 bits for int)
    for (int bit = 0; bit < 32; ++bit) {
        // Map elements to boolean based on current bit
        Common::kernMapToBit<<<blocksPerGrid, BLOCK_SIZE>>>(n, dev_falseBools, dev_input, bit);
        cudaDeviceSynchronize();
        checkCUDAError("kernMapToBit failed!");

        // Flip all of our new booleans (this results in the f array!)
        Common::kernInvert<<<blocksPerGrid, BLOCK_SIZE>>>(n, dev_falseBools);
        cudaDeviceSynchronize();
        checkCUDAError("kernInvert failed!");

        // Scan the f array to get "false" scatter indices
        Efficient::scan(n, dev_falseIndices, dev_falseBools, false);

        // Grab total falses out of last array elements
        int lastBool, lastFalseIndex;
        cudaMemcpy(&lastBool, &dev_falseBools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lastFalseIndex, &dev_falseIndices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
        int totalFalses = lastBool + lastFalseIndex;

        // Scatter elements based on bit value (0s first, then 1s)
        Radix::kernCustomScatter<<<blocksPerGrid, BLOCK_SIZE>>>(
            n, dev_output, dev_input, dev_falseBools, dev_falseIndices, totalFalses);
        cudaDeviceSynchronize();
        checkCUDAError("kernScatterRadix failed!");

        // Swap input and output buffers for next iteration
        int *temp = dev_input;
        dev_input = dev_output;
        dev_output = temp;
    }

    timer().endGpuTimer();

    // Copy result back to host
    cudaMemcpy(odata, dev_input, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_output);
    cudaFree(dev_falseBools);
    cudaFree(dev_falseIndices);
}

} // namespace Radix
} // namespace StreamCompaction