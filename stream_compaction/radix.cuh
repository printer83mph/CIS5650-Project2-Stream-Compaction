#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace StreamCompaction {
namespace Radix {
__global__ void kernScatterRadix(int n, int *odata, const int *idata, const int *bools,
                                 const int *falseIndices, int totalFalses);

}
} // namespace StreamCompaction