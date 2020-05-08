import sys
import argparse
import zipfile

import torch
import numpy as np

from . import reader



parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",  help="bioimage model zip", required=True)
parser.add_argument("image", nargs="?", help="image to process")
parser.add_argument("-o", "--output", nargs="?", help="image to process", required=True)


def main():
    args = parser.parse_args()
    with zipfile.ZipFile(args.model, "r") as model_zip:
        model = reader.eval_model_zip(model_zip, devices=["cpu"])

    input_tensor = np.load(args.image)
    torch_input = torch.from_numpy(input_tensor)
    res = model.forward(torch_input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
