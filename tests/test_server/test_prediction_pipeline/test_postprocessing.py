import numpy as np
import xarray as xr
from pybio.spec.nodes import Postprocessing

from tiktorch.server.prediction_pipeline._postprocessing import make_postprocessing


def test_clip_preprocessing():
    clip_spec = Postprocessing(name="clip", kwargs={"min": 3, "max": 5})
    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    expected = xr.DataArray(
        np.array(
            [
                [3, 3, 3],
                [3, 4, 5],
                [5, 5, 5],
            ]
        ),
        dims=("x", "y"),
    )
    postprocessor = make_postprocessing([clip_spec])
    result = postprocessor(data)
    xr.testing.assert_equal(expected, result)
