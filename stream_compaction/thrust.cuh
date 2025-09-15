#pragma once

#include "common.cuh"

namespace StreamCompaction {
    namespace Thrust {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
    }
}
