#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.cuh"

namespace StreamCompaction {
namespace Radix {

StreamCompaction::Common::PerformanceTimer &timer();

void cpu_sort(int n, int *odata, const int *idata);
void sort(int n, int *odata, const int *idata);

} // namespace Radix
} // namespace StreamCompaction