import argparse

from utils import DataLoader
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--data-dir", type=str, default="./data/")
    parser.add_argument("-n", "--name", type=str, default="ria")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    loader = DataLoader(args.data_dir, args.seed, args.name)
    loader.save()
