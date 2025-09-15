#include <cstdio>
#include "cpu.cuh"

#include "common.cuh"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // Ye olde iterative implementation
            odata[0] = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = idata[i - 1] + odata[i - 1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // Sequentially add up all nonzero input numbers in output array
            int matchingCount = 0;
            for (int j = 0; j < n; ++j) {
                if (idata[j] == 0)
                    continue;

                odata[matchingCount] = idata[j];
                matchingCount++;
            }

            timer().endCpuTimer();
            return matchingCount;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int totalMatchingCount = 0;

            // Populate temp array with 0 if input is 0, 1 otherwise
            int *matchingMask = new int[n];
            int *scannedIndices = new int[n];
            for (int i = 0; i < n; ++i) {
                matchingMask[i] = (bool)idata[i];
                totalMatchingCount++;
            }

            // Run ye olde scan
            scan(n, scannedIndices, matchingMask);

            // Pull from input using scanned indices
            for (int i = 0; i < n; ++i) {
                if (!matchingMask[i])
                    continue;

                odata[scannedIndices[i]] = idata[i];
            }

            delete[] matchingMask;
            delete[] scannedIndices;

            timer().endCpuTimer();
            return totalMatchingCount;
        }
    }
}
