#include "radix.cuh"

namespace StreamCompaction {
namespace Radix {

/**
 *  Runs custom scatter operation for radix sort partitioning.
 *
 * @param falseIndices  An exclusive scan of the `bools` array
 * @param totalFalses   The total number of 1 bits in `bools`
 */
__global__ void kernScatterRadix(int n, int *odata, const int *idata, const int *bools,
                                 const int *falseIndices, int totalFalses) {
    int threadIndex = threadIdx.x + (blockIdx.x * blockDim.x);
    if (threadIndex >= n)
        return;

    int trueIndex = threadIndex - falseIndices[threadIndex] + totalFalses;

    int idataIndex = bools[threadIndex] ? trueIndex : falseIndices[threadIndex];
    odata[threadIndex] = idata[idataIndex];
}

} // namespace Radix
} // namespace StreamCompaction