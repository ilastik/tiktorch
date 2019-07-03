"""
Types defining interop between client and server
"""
# This file should only contain types built on primitives
# available both on ilastik and tiktorch side (e.g. numpy, python stdlib)
import numpy as np
from dataclasses import dataclass

from typing import List, Tuple, Optional, Union, Sequence, NamedTuple


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

    def __init__(
        self, array: np.ndarray, label: np.ndarray, id_: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None
    ) -> None:
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


class Point:
    order: list

    def __init__(self, order: Optional[list] = None, **axes: int):
        if order is None:
            self.order = list(axes.keys())
        else:
            self.order = list(order)

        assert isinstance(self.order, list), self.order
        assert len(set(self.order)) == len(self.order), self.order
        assert len(axes) == len(self.order), (self.order, axes)
        assert all([isinstance(a, str) for a in self.order]), self.order
        for a, v in axes.items():
            assert a in self.order
            setattr(self, a, v)

        super().__init__()

    def __getitem__(self, key: Union[int, str]):
        print("key", key)
        if isinstance(key, int):
            key = self.order[key]

        return getattr(self, key)

    def __setitem__(self, key: Union[int, str], item: int):
        if isinstance(key, int):
            key = self.order[key]

        return setattr(self, key, item)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{a}:{getattr(self, a)}' for a in self.order])})"

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        for a in self.order:
            yield getattr(self, a)

    def __bool__(self):
        return bool(len(self))

    @staticmethod
    def upcast_dims(a: "Point", b: "Point") -> Tuple["Point", "Point"]:
        aa = Point(a.order, **{axis: getattr(a, axis) for axis in a.order})
        bb = Point(b.order, **{axis: getattr(b, axis) for axis in b.order})

        in_a_not_b = [axis for axis in a.order if axis not in b.order]
        in_b_not_a = [axis for axis in b.order if axis not in a.order]

        for axis in in_a_not_b:
            bb.add_dim(axis, 0)

        for axis in in_b_not_a:
            aa.add_dim(axis, 0)

        return aa, bb

    def __lt__(self, other):
        return all([getattr(self, axis, 0) < getattr(other, axis, 0) for axis in set(self.order) | set(other.order)])

    def __gt__(self, other):
        return all([getattr(self, axis, 0) > getattr(other, axis, 0) for axis in set(self.order) | set(other.order)])

    def __eq__(self, other):
        if other is None:
            return False
        elif isinstance(other, (int, float)):
            return all([getattr(self, a) == other for a in self.order])
        else:
            return all(
                [getattr(self, axis, 0) == getattr(other, axis, 0) for axis in set(self.order) | set(other.order)]
            )

    def __le__(self, other):
        if other is None:
            return False
        else:
            return all(
                [getattr(self, axis, 0) <= getattr(other, axis, 0) for axis in set(self.order) | set(other.order)]
            )

    def __ge__(self, other):
        if other is None:
            return (False,)
        else:
            return all(
                [getattr(self, axis, 0) >= getattr(other, axis, 0) for axis in set(self.order) | set(other.order)]
            )

    def __sub__(self, other):
        ret = Point(self.order, **{a: getattr(self, a) for a in self.order})
        for a in other.order:
            if a in self.order:
                setattr(ret, a, getattr(self, a) - getattr(other, a))
            else:
                ret.add_dim(a, -getattr(other, a))

        return ret

    def __add__(self, other):
        ret = Point(self.order, **{a: getattr(self, a) for a in self.order})
        for a in other.order:
            if a in self.order:
                setattr(ret, a, getattr(self, a) + getattr(other, a))
            else:
                ret.add_dim(a, +getattr(other, a))

        return ret

    def __mod__(self, mod: int):
        return Point(self.order, **{a: getattr(self, a) % mod for a in self.order})

    def __floordiv__(self, other: Union[int, float]):
        return Point(self.order, **{a: getattr(self, a) // other for a in self.order})

    def drop_dim(self, *axes: str) -> "Point":
        for a in axes:
            if a in self.__dict__:
                self.order.remove(a)
                del self.__dict__[a]

        return self

    def drop_batch(self) -> "Point":
        self.drop_dim("b")
        return self

    def add_dim(self, name: str, value: int, index: int = -1) -> "Point":
        assert name not in self.order
        setattr(self, name, value)
        self.order.insert(index, name)
        return self

    def add_batch(self, b: int) -> "Point":
        self.add_dim("b", b, index=0)
        return self


class SetDeviceReturnType(NamedTuple):
    training_shape: Tuple[int, ...]
    valid_shapes: List[Tuple[int, ...]]
    shrinkage: Tuple[int, ...]


@dataclass
class ModelState:
    loss: float
    epoch: int
    model_state: bytes
    optimizer_state: bytes
    num_iterations_done: int
    num_iterations_max: int


@dataclass
class Model:
    code: bytes
    config: dict

    def __init__(self, *, code: bytes, config: dict) -> None:
        self.code = code
        self.config = config

    def __bool__(self) -> bool:
        return bool(self.code)


Model.Empty = Model(code=b"", config={})
