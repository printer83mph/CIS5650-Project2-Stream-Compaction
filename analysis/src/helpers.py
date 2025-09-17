from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterator

root_dir = Path.cwd().parent

DEFAULT_BLOCK_SIZE = 256


@dataclass
class BuildParameters:
    enable_scan: bool = False
    enable_compact: bool = False
    enable_radix: bool = False
    block_size: int = 256
    array_size_pow: int = 24


BASE_PARAMETERS = BuildParameters(
    enable_scan=True,
    enable_compact=True,
    enable_radix=True,
    block_size=256,
    array_size_pow=8,
)


def set_build_params(parameters: BuildParameters) -> None:
    with open(root_dir / "src" / "main.cpp", "r+") as main_file:
        content = main_file.read()
        main_file.seek(0)
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("#define TEST_SCAN"):
                lines[i] = f"#define TEST_SCAN {1 if parameters.enable_scan else 0}"
            elif line.strip().startswith("#define TEST_COMPACT"):
                lines[i] = (
                    f"#define TEST_COMPACT {1 if parameters.enable_compact else 0}"
                )
            elif line.strip().startswith("#define TEST_RADIX"):
                lines[i] = f"#define TEST_RADIX {1 if parameters.enable_radix else 0}"
            elif line.strip().startswith("const int SIZE ="):
                lines[i] = f"const int SIZE = 1 << {parameters.array_size_pow};   // feel free to change the size of array"
        main_file.write("\n".join(lines))
        main_file.truncate()

    # Update BLOCK_SIZE wherever it's defined
    for filename in ("naive.cu", "radix.cu"):
        filepath = root_dir / "stream_compaction" / filename
        with open(filepath, "r+") as naive_file:
            content = naive_file.read()
            naive_file.seek(0)
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("#define BLOCK_SIZE"):
                    lines[i] = f"#define BLOCK_SIZE {parameters.block_size}"
            naive_file.write("\n".join(lines))
            naive_file.truncate()


@contextmanager
def use_build_params(parameters: BuildParameters) -> Iterator[None]:
    set_build_params(parameters)
    yield
    set_build_params(BASE_PARAMETERS)


def run_test_pipe_to_file(filename: str):
    with open(root_dir / "analysis/.temp" / f"{filename}.log", "w") as log_file:
        subprocess.call(
            "./bin/cis5650_stream_compaction_test",
            cwd=root_dir / "build",
            stdout=log_file,
            stderr=log_file,
        )


def parse_test_results(filename: str) -> list[tuple[str, float]]:
    log_path = root_dir / "analysis/.temp" / f"{filename}.log"
    results = []

    with open(log_path, "r") as log_file:
        for line in log_file:
            line = line.strip()
            if line.startswith("====") and line.endswith("===="):
                method_name = line[4:-4].strip()
                method_name = method_name.replace(",", "")
            elif line.startswith("elapsed time:") and "ms" in line:
                # Extract time from format "elapsed time: 0.174592ms"
                time_part = line.split("elapsed time:")[1].strip()
                time_str = time_part.split("ms")[0].strip()
                elapsed_time = float(time_str)
                results.append((method_name, elapsed_time))

    return results


def test_with_params(
    *, filename: str, parameters: BuildParameters
) -> list[tuple[str, float]]:
    with use_build_params(parameters):
        subprocess.call(["cmake", "--build", "."], cwd=root_dir / "build")
    run_test_pipe_to_file(filename)
    return parse_test_results(filename)
