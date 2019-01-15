import argparse

from utils import DataLoader
from rouge import Rouge
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--data-dir", type=int, default="./data/")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    loader = DataLoader(args.data_dir)
    loader.save()

    rouge = Rouge()
    scores = rouge.get_scores([val[1] for val in loader.data["test"]], [val[0] for val in loader.data["test"]])
    print(scores)
