import argparse

from parse_samples import parse_samples
from datamodel.test_manager import TestManager


def parse_args():
    parser = argparse.ArgumentParser(description="Parse penguin samples from a CSV file")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument('--blank', type=str, action='append', help="Blank positions as row,col (can be used multiple times)")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = parse_samples(args.input_file)
    print(f"Successfully parsed {len(samples)} samples")

    # Fixed plate dimensions: 8 rows x 12 columns
    ROWS = 8
    COLS = 12
    
    blank_positions = [tuple(map(int, b.split(','))) for b in args.blank] if args.blank else []
    manager = TestManager(samples, ROWS, COLS, blank_positions)
    manager.fill_plates()
    manager.print_plates()
    metrics = manager.algorithm_metrics()
    print("Algorithm Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main() 