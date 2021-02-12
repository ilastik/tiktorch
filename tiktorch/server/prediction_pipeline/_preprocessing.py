from typing import List

import xarray
from pybio.spec.nodes import Preprocessing

from ._types import Transform

ADD_BATCH_DIM = Preprocessing(name="__tiktorch_add_batch_dim", kwargs=None)


def make_ensure_dtype_preprocessing(dtype):
    return Preprocessing(name="__tiktorch_ensure_dtype", kwargs={"dtype": dtype})


def zero_mean_unit_variance(tensor: xarray.DataArray, axes=None, eps=1.0e-6, mode="per_sample") -> xarray.DataArray:
    if axes:
        axes = tuple(axes)
        mean, std = tensor.mean(axes), tensor.std(axes)
    else:
        mean, std = tensor.mean(), tensor.std()

    if mode != "per_sample":
        raise NotImplementedError(f"Unsupported mode for zero_mean_unit_variance: {mode}")

    return (tensor - mean) / (std + 1.0e-6)


def ensure_dtype(tensor: xarray.DataArray, *, dtype):
    """
    Convert array to a given datatype
    """
    return tensor.astype(dtype)


def add_batch_dim(tensor: xarray.DataArray):
    """
    Add a singleton batch dimension
    """
    return tensor.expand_dims("b")


KNOWN_PREPROCESSING = {
    "zero_mean_unit_variance": zero_mean_unit_variance,
    "__tiktorch_add_batch_dim": add_batch_dim,
    "__tiktorch_ensure_dtype": ensure_dtype,
}


def chain(*functions):
    def _chained_function(tensor):
        tensor = tensor
        for fn, kwargs in functions:
            kwargs = kwargs or {}
            tensor = fn(tensor, **kwargs)

        return tensor

    return _chained_function


def make_preprocessing(preprocessing_spec: List[Preprocessing]) -> Transform:
    """
    :param preprocessing: bioimage-io spec node
    """
    preprocessing_functions = []

    step: Preprocessing
    for step in preprocessing_spec:
        fn = KNOWN_PREPROCESSING.get(step.name)
        kwargs = step.kwargs.copy() if step.kwargs else {}

        if fn is None:
            raise NotImplementedError(f"Preprocessing {step.name}")

        preprocessing_functions.append((fn, kwargs))

    return chain(*preprocessing_functions)
