#include "common.cuh"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            int threadIndex = threadIdx.x + (blockIdx.x * blockDim.x);
            if (threadIndex >= n)
                return;

            // Cast to boolean to check against 0
            bools[threadIndex] = (bool)idata[threadIndex];
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            int threadIndex = threadIdx.x + (blockIdx.x * blockDim.x);
            if (threadIndex >= n)
                return;

            if (bools[threadIndex]) {
                odata[indices[threadIndex]] = idata[threadIndex];
            }
        }

        /**
         * Similar to kernMapToBoolean, but the result (array of bools) is built from the kth bits
         * of input array elements. Bits are 0-indexed, i.e. kth power of 2.
         */
        __global__ void kernMapToBit(int n, int *bools, const int *idata, int bit) {
            int threadIndex = threadIdx.x + (blockIdx.x * blockDim.x);
            if (threadIndex >= n)
                return;

            int bitValue = (idata[threadIndex] >> bit) & 1;
            bools[threadIndex] = bitValue;
        }
    }
}
