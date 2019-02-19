"""
Types defining interop between client and server
"""
# This file should only contain types built on primitives
# available both on ilastik and tiktorch side (e.g. numpy, python stdlib)
from typing import List, Tuple, Optional

import numpy as np


class NDArray:
    """
    Containter for numpy array to transfer additional properties
    e.g. position of array in dataset (id_)
    Numpy array
    """

    # TODO: Maybe convert id_ from Optional[Tuple[int, ...]] to str
    def __init__(
        self,
        array: np.ndarray,
        id_: Optional[Tuple[int, ...]] = None
    ) -> None:
        self._numpy = array
        self.id = id_

    def as_numpy(self) -> np.ndarray:
        return self._numpy

    @property
    def dtype(self):
        return self._numpy.dtype

    @property
    def shape(self):
        return self._numpy.shape


class NDArrayBatch:
    """
    Batch of NDArrays
    """
    def __init__(self, arrays: List[NDArray]):
        self._arrays = arrays

    def array_metas(self):
        return [{
            'dtype': arr.dtype.str,
            'shape': arr.shape,
            'id': arr.id,
        } for arr in self._arrays]

    def __len__(self):
        return len(self._arrays)

    def __iter__(self):
        for item in self._arrays:
            yield item

    def as_numpy(self) -> List[np.ndarray]:
        return [
            arr.as_numpy() for arr in self._arrays
        ]
