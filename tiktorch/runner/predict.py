import argparse
import sys
import zipfile

import numpy as np
import xarray as xr

from tiktorch.runner import utils

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="bioimage model zip", required=True)
parser.add_argument("image", nargs="?", help="image to process (.npy)")
parser.add_argument("-o", "--output", nargs="?", help="output image (.npy)", required=True)


def main():
    args = parser.parse_args()
    with zipfile.ZipFile(args.model, "r") as model_zip:
        model = utils.eval_model_zip(model_zip, devices=["cpu"])

    input_tensor = np.load(args.image)
    tagged_data = xr.DataArray(input_tensor, dims=tuple(model.input_axes))
    res = model.forward(tagged_data)
    np.save(args.output, res.data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
