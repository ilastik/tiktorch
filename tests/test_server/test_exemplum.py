from pathlib import Path

import numpy
import torch

from pybio.spec import load_spec_and_kwargs
from tiktorch.server.exemplum import Exemplum


def test_exemplum(datadir, cache_path):

    spec_path = Path(datadir) / "unet2d/UNet2DNucleiBroad.model.yaml"
    assert spec_path.exists()
    pybio_model = load_spec_and_kwargs(str(spec_path), cache_path=cache_path)

    exemplum = Exemplum(pybio_model=pybio_model, _devices=[torch.device("cpu")], warmstart=True)
    test_ipt = numpy.load(pybio_model.spec.test_input)
    out_seq = exemplum.forward(test_ipt)
    assert isinstance(out_seq, (list, tuple))
    assert len(out_seq) == 1
    out = out_seq[0]
    expected_out = numpy.load(pybio_model.spec.test_output)
    assert numpy.isclose(out, expected_out).all()
