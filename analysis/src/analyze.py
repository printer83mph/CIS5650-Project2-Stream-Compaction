from collections import defaultdict

import helpers
from pathlib import Path


def test_optimize_block_sizes():
    block_size_and_runtime_by_algorithm: defaultdict[str, list[tuple[int, int]]] = (
        defaultdict(list)
    )

    for block_size_exp in range(4, 11):
        block_size = pow(2, block_size_exp)

        results = helpers.test_with_params(
            filename="temp",
            parameters=helpers.BuildParameters(
                enable_scan=True,
                enable_compact=True,
                enable_radix=True,
                block_size=block_size,
                array_size_pow=20,
            ),
        )

        for algorithm, runtime in results:
            block_size_and_runtime_by_algorithm[algorithm].append((block_size, runtime))

    # Print CSV header
    algorithms = list(block_size_and_runtime_by_algorithm.keys())

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
