from pathlib import Path

import numpy
import torch
from pybio.spec import load_spec_and_kwargs

from tiktorch.server.exemplum import Exemplum


def test_exemplum(data_path, cache_path):
    spec_path = data_path / "unet2d/UNet2DNucleiBroad.model.yaml"
    assert spec_path.exists(), spec_path.absolute()
    pybio_model = load_spec_and_kwargs(str(spec_path), cache_path=cache_path)

    exemplum = Exemplum(pybio_model=pybio_model, _devices=[torch.device("cpu")])
    test_ipt = numpy.load(pybio_model.spec.test_input)  # test input with batch dim
    out = exemplum.forward(test_ipt[0])  # todo: exemplum.forward should get batch with batch dim
    # assert isinstance(out_seq, (list, tuple)) # todo: forward should return a list
    # assert len(out_seq) == 1
    # out = out_seq
    expected_out = numpy.load(pybio_model.spec.test_output)[
        0
    ]  # todo: remove [0], exemplum.forward should return batch with batch dim
    assert numpy.isclose(out, expected_out).all()
