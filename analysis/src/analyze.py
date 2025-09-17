from collections import defaultdict

import helpers
from pathlib import Path


def test_optimize_block_sizes():
    block_size_and_runtime_by_algorithm: defaultdict[str, list[tuple[int, int]]] = (
        defaultdict(list)
    )

    for block_size_exp in range(5, 11):
        block_size = pow(2, block_size_exp)

        results = helpers.test_with_params(
            filename="temp",
            parameters=helpers.BuildParameters(
                enable_scan=True,
                enable_compact=True,
                enable_radix=True,
                block_size=block_size,
                array_size_pow=24,
            ),
        )

        for algorithm, runtime in results:
            block_size_and_runtime_by_algorithm[algorithm].append((block_size, runtime))

    # Print CSV header
    algorithms = list(block_size_and_runtime_by_algorithm.keys())
    algorithms = [alg for alg in algorithms if "cpu" not in alg.lower()]

    # Get all block sizes (assuming they're the same for all algorithms)
    block_sizes = [
        pair[0] for pair in block_size_and_runtime_by_algorithm[algorithms[0]]
    ]

    # Create reports directory if it doesn't exist
    output_path = Path.cwd() / "reports/block_sizes.csv"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        # Write CSV header
        f.write("Block Size," + ",".join(algorithms) + "\n")

        # Print each row
        for i, block_size in enumerate(block_sizes):
            row = [str(block_size)]
            for algorithm in algorithms:
                runtime = block_size_and_runtime_by_algorithm[algorithm][i][1]
                row.append(str(runtime))
            f.write(",".join(row) + "\n")


def compare_scan_implementations():
    array_size_and_runtime_by_algorithm: defaultdict[str, list[tuple[int, int]]] = (
        defaultdict(list)
    )

    for array_size_exp in range(18, 29):
        array_size = pow(2, array_size_exp)

        results = helpers.test_with_params(
            filename="temp",
            parameters=helpers.BuildParameters(
                enable_scan=True,
                enable_compact=False,
                enable_radix=False,
                block_size=128,
                array_size_pow=array_size_exp,
            ),
        )

        for algorithm, runtime in results:
            array_size_and_runtime_by_algorithm[algorithm].append((array_size, runtime))

    # Create reports directory if it doesn't exist
    output_path = Path.cwd() / "reports/scan_comparison.csv"
    output_path.parent.mkdir(exist_ok=True)

    # Get algorithms and array sizes
    algorithms = list(array_size_and_runtime_by_algorithm.keys())
    array_sizes = [
        pair[0] for pair in array_size_and_runtime_by_algorithm[algorithms[0]]
    ]

    with open(output_path, "w") as f:
        # Write CSV header
        f.write("Array Size," + ",".join(algorithms) + "\n")

        # Write each row
        for i, array_size in enumerate(array_sizes):
            row = [str(array_size)]
            for algorithm in algorithms:
                runtime = array_size_and_runtime_by_algorithm[algorithm][i][1]
                row.append(str(runtime))
            f.write(",".join(row) + "\n")
