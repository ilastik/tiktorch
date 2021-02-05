import argparse
import os
import sys
import shutil
import tempfile
import warnings

from pathlib import Path
from zipfile import ZipFile, BadZipFile

import numpy as np
from numpy.testing import assert_array_almost_equal
from pybio import spec

from .reader import guess_model_path
from .model_adapter import create_model_adapter

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="bioimage model.yaml or model.zip", required=True)
parser.add_argument("-f", "--format", help="weight format to use for the test", default='')
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


def main():
    args = parser.parse_args()
    # try opening model from model.zip
    try:
        with ZipFile(args.model, "r") as model_zip:
            pybio_model, cache_path = _load_from_zip(model_zip)
    # otherwise open from model.yaml
    except BadZipFile:
        spec_path = os.path.abspath(args.model)
        pybio_model = spec.load_and_resolve_spec(spec_path)
        cache_path = None

    weight_format = args.format if args.format else None
    model = create_model_adapter(pybio_model=pybio_model, devices=['cpu'], weight_format=weight_format)

    test_inputs = pybio_model.test_inputs
    test_outputs = pybio_model.test_outputs

    for test_input, test_output in zip(test_inputs, test_outputs):
        input_tensor = np.load(test_input).astype('float32')
        exp = np.load(test_output)
        res = model.forward(input_tensor)
        assert_array_almost_equal(exp, res, args.decimals)

    if cache_path is not None:

        def _on_error(function, path, exc_info):
            warnings.warn("Failed to delete temp directory %s", path)
        shutil.rmtree(cache_path, onerror=_on_error)

    print(f"All results match the expected output for {len(test_inputs)} test datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
