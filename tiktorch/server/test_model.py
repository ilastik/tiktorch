import argparse
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Union
from zipfile import BadZipFile, ZipFile

import numpy as np
import xarray as xr
from bioimageio import spec
from numpy.testing import assert_array_almost_equal

from .prediction_pipeline import create_prediction_pipeline, get_weight_formats
from .reader import guess_model_path

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="bioimage model.yaml or model.zip", required=True)
parser.add_argument(
    "-f", "--weight-format", help="weight format to use for the test", default=None, choices=get_weight_formats()
)
parser.add_argument("-d", "--decimals", help="test precision", default=4, type=float)


def _load_from_zip(model_zip: ZipFile):
    temp_path = Path(tempfile.mkdtemp(prefix="tiktorch_"))
    cache_path = temp_path / "cache"

    model_zip.extractall(temp_path)

    spec_file_str = guess_model_path([str(file_name) for file_name in temp_path.glob("*")])
    if not spec_file_str:
        raise Exception(
            "Model config file not found, make sure that .model.yaml file in the root of your model archive"
        )
    return spec.load_and_resolve_spec(spec_file_str), cache_path


def load_data(path_to_npy: Union[Path, str], spec):
    return xr.DataArray(np.load(path_to_npy), dims=tuple(spec.axes))


def main():
    args = parser.parse_args()
    # try opening model from model.zip
    try:
        with ZipFile(args.model, "r") as model_zip:
            bioimageio_model, cache_path = _load_from_zip(model_zip)
    # otherwise open from model.yaml
    except BadZipFile:
        spec_path = os.path.abspath(args.model)
        bioimageio_model = spec.load_and_resolve_spec(spec_path)
        cache_path = None

    model = create_prediction_pipeline(
        bioimageio_model=bioimageio_model, devices=["cpu"], weight_format=args.weight_format, preserve_batch_dim=True
    )

    input_args = [
        load_data(inp, inp_spec) for inp, inp_spec in zip(bioimageio_model.test_inputs, bioimageio_model.inputs)
    ]
    expected_outputs = [
        load_data(out, out_spec) for out, out_spec in zip(bioimageio_model.test_outputs, bioimageio_model.outputs)
    ]

    results = [model.forward(*input_args)]

    for res, exp in zip(results, expected_outputs):
        assert_array_almost_equal(exp, res, args.decimals)

    if cache_path is not None:

        def _on_error(function, path, exc_info):
            warnings.warn("Failed to delete temp directory %s", path)

        shutil.rmtree(cache_path, onerror=_on_error)

    print("All results match the expected output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
