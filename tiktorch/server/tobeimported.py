import numpy
import torch
from inferno.io.transform import Transform
from scipy.ndimage import convolve


# from neurofire:
# todo: move to inferno (or not) and import
class DtypeMapping(object):
    DTYPE_MAPPING = {
        "float32": "float32",
        "float": "float32",
        "double": "float64",
        "float64": "float64",
        "half": "float16",
        "float16": "float16",
    }
    INVERSE_DTYPE_MAPPING = {"float32": "float", "float64": "double", "float16": "half", "int64": "long"}


class Segmentation2Membranes(Transform, DtypeMapping):
    """Convert dense segmentation to boundary-maps (or membranes)."""

    def __init__(self, dtype="float32", **super_kwargs):
        super(Segmentation2Membranes, self).__init__(**super_kwargs)
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dtype = self.DTYPE_MAPPING.get(dtype)

    def image_function(self, image):
        if isinstance(image, numpy.ndarray):
            return self._apply_numpy_tensor(image)
        elif torch.is_tensor(image):
            return self._apply_torch_tensor(image)
        else:
            raise NotImplementedError("Only support np.ndarray and torch.tensor, got %s" % type(image))

    def _apply_numpy_tensor(self, image):
        gx = convolve(numpy.float32(image), numpy.array([-1.0, 0.0, 1.0]).reshape(1, 3))
        gy = convolve(numpy.float32(image), numpy.array([-1.0, 0.0, 1.0]).reshape(3, 1))
        return getattr(numpy, self.dtype)((gx**2 + gy**2) > 0)
