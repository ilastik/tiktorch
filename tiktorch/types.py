"""
Types defining interop between client and server
"""
# This file should only contain types built on primitives
# available both on ilastik and tiktorch side (e.g. numpy, python stdlib)
from typing import List, Tuple, Optional, Union

import numpy as np


class NDArray:
    """
    Containter for numpy array to transfer additional properties
    e.g. position of array in dataset (id_)
    Numpy array
    """

    # TODO: Maybe convert id_ from Optional[Tuple[int, ...]] to str
    def __init__(self, array: np.ndarray, id_: Optional[Tuple[int, ...]] = None) -> None:
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

    def add_label(self, label: np.ndarray) -> "LabeledNDArray":
        return LabeledNDArray(array=self._numpy, label=label, id_=self.id)


class LabeledNDArray(NDArray):
    """
    Containter for numpy array with a label and an id
    """

    def __init__(self, array: np.ndarray, label: np.ndarray, id_: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None) -> None:
        super().__init__(array, id_)
        self._label = label

    def as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._numpy, self._label

    def drop_label(self) -> NDArray:
        return NDArray(array=self._numpy, id_=self.id)


class NDArrayBatch:
    """
    Batch of NDArrays
    """

    def __init__(self, arrays: List[NDArray]):
        self._arrays = arrays

    def array_metas(self):
        return [{"dtype": arr.dtype.str, "shape": arr.shape, "id": arr.id} for arr in self._arrays]

    def __len__(self):
        return len(self._arrays)

    def __iter__(self):
        for item in self._arrays:
            yield item

    def as_numpy(self) -> List[np.ndarray]:
        return [arr.as_numpy() for arr in self._arrays]


class LabeledNDArrayBatch(NDArrayBatch):
    """
    Batch of LabeledNDArrays
    """
    def __init__(self, arrays: List[LabeledNDArray]):
        super().__init__(arrays)


# class DataPoint():
#     order: str = 'tczyx'
#     t: int
#     c: int
#     z: int
#     y: int
#     x: int
#
#     def __init__(self, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
#         self.t = t
#         self.c = c
#         self.z = z
#         self.y = y
#         self.x = x
#
#     def __getitem__(self, key: Union[int, str]):
#         if isinstance(key, int):
#             key = self.order[key]
#
#         return getattr(self, key)
#
#     def __setitem__(self, key: Union[int, str], item: int):
#         if isinstance(key, int):
#             key = self.order[key]
#
#         return setattr(self, key, item)
#
#     def __repr__(self):
#         return ', '.join([f"{a}:{getattr(self, a)}" for a in self.order])
#
#     def __len__(self):
#         return len(self.order)
#
#     def __iter__(self):
#         for a in self.order:
#             yield getattr(self, a)
#
#     def as_2d(self, dp) -> "DataPoint2D"
#         self.__class__ = DataPoint2D
#         self.order = DataPoint2D.order
#         return self
#
# class DataPoint2D(DataPoint):
#     order: str = 'cyx'
#
#     def __init__(self, c: int = 0, y: int = 0, x: int = 0):
#         super().__init__(c=c, y=y, x=x)
#
#
#
# class DataPoint3D(DataPoint2D):
#     order: str = 'czyx'
#
#     def __init__(self, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
#         self.z: int = z
#         super().__init__(c=c, y=y, x=x)
#
# class DataPoint(DataPoint3D):
#     order: str = 'tczyx'
#
#         self.t: int = t
#         super().__init__(c=c, z=z, y=y, x=x)
#
# class BatchBase:
#     b: int = 0
#
# class DataPointBatch(DataPoint, BatchBase):
#     pass
#
#
#
# c, y, x = dp
# class DataPoint(NamedTuple):
#     """ A point in the ilastik standard tczyx data representation """
#
#     t: int = 1
#     c: int = 1
#     z: int = 1
#     y: int = 1
#     x: int = 1
# #
# dt_with_t = DataPoint()
# dt_without_t = DataPoint3D(dt)
# class DataPoint3D(NamedTuple):
#     """ czyx """
#
#     c: int = 1
#     z: int = 1
#     y: int = 1
#     x: int = 1
#
# class DataPoint2D(NamedTuple):
#     """ cyx """
#
#     c: int = 1
#     y: int = 1
#     x: int = 1
#
#
# class BatchDataPointBase(NamedTuple):
#     b: int
#
#
# class BatchDataPoint(BatchDataPointBase, DataPoint):
#     """ btczyx """
#
#
# class BatchDataPoint3D(BatchDataPointBase, DataPoint3D):
#     """ bczyx """
#
#
# class BatchDataPoint2D(BatchDataPointBase, DataPoint2D):
#     """ bcyx """
#
#
# dp2d = DataPoint2D(3, 5)
# dp = DataPoint(*dp2d)
#
#
#
# from collections import namedtuple
#
# class DataPoint1D():
#     order = 'cx'
#     def __init__(self, c: int = 0, x: int = 0):
#         self.c: int = c
#         self.x: int = x
#
#     def __getitem__(self, key: Union[int, str]):
#         if isinstance(key, int):
#             key = self.order[key]
#
#         return getattr(self, key)
#
#     def __setitem__(self, key: Union[int, str], item: int):
#         if isinstance(key, int):
#             key = self.order[key]
#
#         return setattr(self, key, item)
#
#     def __repr__(self):
#         return ', '.join([f"{a}:{getattr(self, a)}" for a in self.order])
#
#     def __len__(self):
#         return len(self.order)
#
#     def __iter__(self):
#         for a in self.order:
#             yield getattr(self, a)
#
# class DataPoint2D(DataPoint1D):
#     order = 'cyx'
#
#     def __init__(self, c: int = 0, y: int = 0, x: int = 0):
#         super().__init__(c=c, x=x)
#         self.y: int = y
# class BatchDataPoint(NamedTuple):
#     """ btczyx """
#     b: int
#     t: DataPoint.t
#     c: DataPoint.c
#     z: DataPoint.z
#     y: DataPoint.y
#     x: DataPoint.x
#
#
# class BatchDataPoint3D(NamedTuple):
#     """ bczyx """
#     b: int
#     c: DataPoint.c
#     z: DataPoint.z
#     y: DataPoint.y
#     x: DataPoint.x
#
#
# class BatchDataPoint2D(NamedTuple):
#     """ bcyx """
#     b: int
#     c: DataPoint.c
#     y: DataPoint.y
#     x: DataPoint.x
