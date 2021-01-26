import argparse
import sys
import zipfile

import numpy as np
import torch

from . import reader

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="bioimage model zip", required=True)
parser.add_argument("image", nargs="?", help="image to process (.npy)")
parser.add_argument("-o", "--output", nargs="?", help="output image (.npy)", required=True)


def main():
    args = parser.parse_args()
    with zipfile.ZipFile(args.model, "r") as model_zip:
        model = reader.eval_model_zip(model_zip, devices=["cpu"])

    input_tensor = np.load(args.image)
    res = model.forward(input_tensor)
    np.save(args.output, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
