import sys
import argparse
from parse_samples import parse_samples
from datamodel.test_manager import TestManager


def parse_args():
    parser = argparse.ArgumentParser(description="Parse penguin samples from a CSV file")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--rows", type=int, help="Number of rows for the assay plate")
    parser.add_argument("--cols", type=int, help="Number of columns for the assay plate")
    parser.add_argument('--blank', type=str, action='append', help="Blank positions as row,col (can be used multiple times)")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = parse_samples(args.input_file)
    print(f"Successfully parsed {len(samples)} samples")

    blank_positions = [tuple(map(int, b.split(','))) for b in args.blank] if args.blank else []
    manager = TestManager(samples, args.rows, args.cols, blank_positions)
    print(f"Number of assay plates needed: {len(manager.get_plates())}")
    manager.print_plates()
    metrics = manager.algorithm_metrics()
    print("Algorithm Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main() 