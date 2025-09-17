# Performance Analysis

## Usage

To get set up, first install `uv`. Sync dependencies with `uv sync`.
Ensure that your build is also set up properly in the `build` folder.

Then, to run a block size comparisons, with `analysis` as working directory, run:

```sh
uv run src/main.py block-sizes
```

Or, alternatively, for scan algorithm comparisons:

```sh
uv run src/main.py scan-comparison
```

## Behavior

This module works by temporarily editing constants in source files, building, and running the executable. It modifies constants like `#define TEST_SCAN 1` to enable or disable testing certain features, and modifies others like `#define BLOCK_SIZE 256` to tweak performance.

It is built to be extensible with more tests.