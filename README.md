CUDA Stream Compaction
======================

![](img/graphs/scan_performance_nonpow2.png)

(why is thrust so slow???)

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Thomas Shaw
  * [LinkedIn](https://www.linkedin.com/in/thomas-shaw-54468b222), [personal website](https://tlshaw.me), [GitHub](https://github.com/printer83mph), etc.
* Tested on: Fedora 42, Ryzen 7 5700x @ 4.67GHz, 32GB, RTX 2070 8GB


## Features

- CUDA exclusive scan and compaction implementations!
- Work-efficient algorithm speeds up scan even more!
- Faster scan than CPU!!
- GPU Radix Sort implementation
  - (tested against CPU version in `main`)


## Performance Analysis

Python scripts have been created in `analysis/` for easier stat collection. The [README](./analysis/README.md) within provides info on how to run these.

### Block Size Optimizations

Let's take a look at the resulting performance from block size choices, all normalized:

![](img/graphs/block_sizes_all_normalized.png)

Performance seems to fluctuate differently per-algorithm between possible block sizes from 64 and 1024. 

See below all the different algorithms independently:
| | | |
|---|---|---|
|  ![](img/graphs/block_sizes_scan_naive.png) Naive Scan | ![](img/graphs/block_sizes_scan_work_efficient.png) Work Efficient Scan | ![](img/graphs/block_sizes_scan_thrust.png) Thrust Scan |
|  | ![](img/graphs/block_sizes_compact.png) Stream Compaction (Work-Efficient Scan) | ![](img/graphs/block_sizes_radix.png) Radix |

It seems that the optimal block size for this machine is somewhere between 128 and 512, but that really depends on the algorithm.