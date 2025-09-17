import sys
import analyze


def main():
    match sys.argv[1].lower():
        case "block-sizes":
            print("running block size optimization analysis...")
            analyze.test_optimize_block_sizes()
            print("done! results stored to reports/block_sizes.csv.")


if __name__ == "__main__":
    main()
