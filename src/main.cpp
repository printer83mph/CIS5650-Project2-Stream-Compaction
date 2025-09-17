/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include "testing_helpers.hpp"
#include <cstdio>
#include <stream_compaction/cpu.cuh>
#include <stream_compaction/efficient.cuh>
#include <stream_compaction/naive.cuh>
#include <stream_compaction/radix.cuh>
#include <stream_compaction/thrust.cuh>

// Control which algorithms to test!
#define TEST_SCAN 1
#define TEST_COMPACT 1
#define TEST_RADIX 1

const int SIZE = 1 << 8;   // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *sourceData = new int[SIZE];
int *referenceResult = new int[SIZE];
int *referenceResultNPOT = new int[SIZE];
int *result = new int[SIZE];

int main(int argresult, char *argv[]) {
    // Scan tests

#if TEST_SCAN

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, sourceData, 50); // Leave a 0 at the end to test that edge case
    sourceData[SIZE - 1] = 0;
    printArray(SIZE, sourceData, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, referenceResult);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, referenceResult, sourceData);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, referenceResult, true);

    zeroArray(SIZE, referenceResultNPOT);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, referenceResultNPOT, sourceData);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, referenceResultNPOT, true);
    printCmpResult(NPOT, referenceResultNPOT, referenceResultNPOT);

    zeroArray(SIZE, result);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, result, sourceData);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(SIZE, result, true);
    printCmpResult(SIZE, referenceResult, result);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, result, a);
    printArray(SIZE, result, true); */

    zeroArray(SIZE, result);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, result, sourceData);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(SIZE, result, true);
    printCmpResult(NPOT, referenceResultNPOT, result);

    zeroArray(SIZE, result);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, result, sourceData);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(SIZE, result, true);
    printCmpResult(SIZE, referenceResult, result);

    zeroArray(SIZE, result);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, result, sourceData);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(NPOT, result, true);
    printCmpResult(NPOT, referenceResultNPOT, result);

    zeroArray(SIZE, result);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, result, sourceData);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(SIZE, result, true);
    printCmpResult(SIZE, referenceResult, result);

    zeroArray(SIZE, result);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, result, sourceData);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(NPOT, result, true);
    printCmpResult(NPOT, referenceResultNPOT, result);

#endif
#if TEST_COMPACT

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, sourceData, 4); // Leave a 0 at the end to test that edge case
    sourceData[SIZE - 1] = 0;
    printArray(SIZE, sourceData, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, referenceResult);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, referenceResult, sourceData);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, referenceResult, true);
    printCmpLenResult(count, expectedCount, referenceResult, referenceResult);

    zeroArray(SIZE, referenceResultNPOT);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, referenceResultNPOT, sourceData);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, referenceResultNPOT, true);
    printCmpLenResult(count, expectedNPOT, referenceResultNPOT, referenceResultNPOT);

    zeroArray(SIZE, result);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, result, sourceData);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, result, true);
    printCmpLenResult(count, expectedCount, referenceResult, result);

    zeroArray(SIZE, result);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, result, sourceData);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(count, result, true);
    printCmpLenResult(count, expectedCount, referenceResult, result);

    zeroArray(SIZE, result);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, result, sourceData);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(count, result, true);
    printCmpLenResult(count, expectedNPOT, referenceResultNPOT, result);

#endif
#if TEST_RADIX

    printf("\n");
    printf("*****************************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("*****************************\n");

    genArray(SIZE - 1, sourceData, 50); // Leave a 0 at the end to test that edge case
    // a[SIZE - 1] = 0;
    printArray(SIZE, sourceData, true);

    zeroArray(SIZE, referenceResult);
    printDesc("cpu sort, power-of-two");
    StreamCompaction::Radix::cpu_sort(SIZE, referenceResult, sourceData);
    printElapsedTime(StreamCompaction::Radix::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(SIZE, referenceResult, true);

    zeroArray(NPOT, result);
    printDesc("cpu sort, non-power-of-two");
    StreamCompaction::Radix::cpu_sort(NPOT, referenceResultNPOT, sourceData);
    printElapsedTime(StreamCompaction::Radix::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(NPOT, referenceResultNPOT, true);

    zeroArray(SIZE, result);
    printDesc("gpu radix sort, power-of-two");
    StreamCompaction::Radix::sort(SIZE, result, sourceData);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(SIZE, result, true);
    printCmpResult(SIZE, referenceResult, result);

    zeroArray(NPOT, result);
    printDesc("gpu radix sort, non-power-of-two");
    StreamCompaction::Radix::sort(NPOT, result, sourceData);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(NPOT, result, true);
    printCmpResult(NPOT, referenceResultNPOT, result);

#endif

#ifdef _WIN32
    system("pause"); // stop Win32 console from closing on exit
#endif
    delete[] sourceData;
    delete[] referenceResult;
    delete[] referenceResultNPOT;
    delete[] result;
}
