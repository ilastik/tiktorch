import numpy
import torch
from pybio.spec.utils.transformers import load_and_resolve_spec, load_spec

from tiktorch.server.model_adapter._exemplum import Exemplum


def test_exemplum(data_path, cache_path):
    spec_path = data_path / "unet2d/UNet2DNucleiBroad.model.yaml"
    assert spec_path.exists(), spec_path.absolute()
    pybio_model = load_and_resolve_spec(str(spec_path))

    exemplum = Exemplum(pybio_model=pybio_model, devices=["cpu"])
    test_ipt = numpy.load(pybio_model.test_inputs[0]).reshape((1, 1, 512, 512))

    out = exemplum.forward(test_ipt)  # todo: exemplum.forward should get batch with batch dim
    # assert isinstance(out_seq, (list, tuple)) # todo: forward should return a list
    # assert len(out_seq) == 1
    # out = out_seq
    expected_out = numpy.load(pybio_model.test_outputs[0])[
        0
    ]  # todo: remove [0], exemplum.forward should return batch with batch dim
    assert numpy.isclose(out, expected_out).all()
