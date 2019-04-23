"""
Types defining interop between client and server
"""
# This file should only contain types built on primitives
# available both on ilastik and tiktorch side (e.g. numpy, python stdlib)
import numpy as np

from collections.abc import Mapping
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


class Point(Mapping):
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
        if isinstance(key, int):
            key = self.order[key]

        return getattr(self, key)

    def __setitem__(self, key: Union[int, str], item: int):
        if isinstance(key, int):
            key = self.order[key]

        return setattr(self, key, item)

    def __repr__(self):
        try:
            return f"{self.__class__.__name__}({', '.join([f'{a}:{getattr(self, a)}' for a in self.order])})"
        except Exception:
            print('here', self.order)
            raise

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
            return all([getattr(self, axis, 0) <= getattr(other, axis, 0) for axis in set(self.order) | set(other.order)])

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

# deprecated:
class PointAndBatchPointBase:
    order: str
    b: int
    t: int
    c: int
    z: int
    y: int
    x: int

    def __init__(self, order: str, **axes: int):
        self.order = order
        assert len(axes) == len(self.order)
        for a, v in axes.items():
            assert a in self.order
            setattr(self, a, v)

        super().__init__()

    def __getitem__(self, key: Union[int, str]):
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
    def upcast_dim(
        a: "PointAndBatchPointBase", b: "PointAndBatchPointBase"
    ) -> Tuple["PointAndBatchPointBase", "PointAndBatchPointBase"]:
        space_dim_a = len(a) - 1
        if a.__class__.__name__.startswith("Batch"):
            space_dim_a -= 1

        space_dim_b = len(b) - 1
        if b.__class__.__name__.startswith("Batch"):
            space_dim_b -= 1

        if space_dim_a < space_dim_b:
            a = a.as_d(space_dim_b)
        elif space_dim_a > space_dim_b:
            b = b.as_d(space_dim_a)

        return a, b

    def __lt__(self, other):
        me, other = self.upcast_dim(self, other)
        return all([m < o for m, o in zip(me, other)])

    def __gt__(self, other):
        me, other = self.upcast_dim(self, other)
        return all([m > o for m, o in zip(me, other)])

    def __eq__(self, other):
        if other is None:
            return False
        elif isinstance(other, (int, float)):
            return all([self[a] == other for a in self.order])
        else:
            me, other = self.upcast_dim(self, other)
            return all([m == o for m, o in zip(me, other)])

    def __le__(self, other):
        me, other = self.upcast_dim(self, other)
        return all([m <= o for m, o in zip(me, other)])

    def __ge__(self, other):
        me, other = self.upcast_dim(self, other)
        return all([m >= o for m, o in zip(me, other)])

    def __sub__(self, other):
        me, other = self.upcast_dim(self, other)
        return me.__class__(**{a: me[a] - other[a] for a in me.order})

    def __add__(self, other):
        me, other = self.upcast_dim(self, other)
        return me.__class__(**{a: me[a] + other[a] for a in me.order})

    def __mod__(self, mod: int):
        return self.__class__(**{a: self[a] % mod for a in self.order})

    def __floordiv__(self, other: Union[int, float]):
        return self.__class__(**{a: self[a] // other for a in self.order})

    def drop_batch(self):
        raise NotImplementedError("To be implemented in subclass!")


class BatchPointBase(PointAndBatchPointBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_spacetime(
        b: int, c: int, spacetime: Sequence[int]
    ) -> Union["BatchPoint2D", "BatchPoint3D", "BatchPoint4D"]:
        """
        :return: a suitable BatchPoint instance
        :raises: ValueError
        """
        if len(spacetime) == 4:
            t, z, y, x = spacetime
            return BatchPoint4D(b, t, c, z, y, x)
        elif len(spacetime) == 3:
            return BatchPoint3D(b, c, *spacetime)
        elif len(spacetime) == 2:
            return BatchPoint2D(b, c, *spacetime)
        else:
            raise ValueError(f"Uninterpretable spacetime: {spacetime}")

    def as_2d(self) -> "BatchPoint2D":
        return BatchPoint2D(b=self.b, c=self.c, y=self.y, x=self.x)

    def as_3d(self) -> "BatchPoint3D":
        return BatchPoint3D(b=self.b, c=self.c, z=self.z, y=self.y, x=self.x)

    def as_4d(self) -> "BatchPoint4D":
        return BatchPoint4D(b=self.b, t=self.t, c=self.c, z=self.z, y=self.y, x=self.x)


class BatchPoint(BatchPointBase):
    def __init__(self, order: str, b: int, **kwargs):
        assert all([a in order for a in kwargs])
        assert all([a in kwargs for a in order])
        self.order = order
        super().__init__(b=b, **kwargs)

    def drop_batch(self):
        return Point(self.order, **{a: getattr(self, a) for a in self.order.replace("b", "")})


class BatchPoint2D(BatchPointBase):
    order: str = "bcyx"

    def __init__(self, b: int, c: int, y: int, x: int):
        super().__init__(b=b, c=c, y=y, x=x)

    def drop_batch(self) -> "Point2D":
        return Point2D(c=self.c, y=self.y, x=self.x)


class BatchPoint3D(BatchPointBase):
    order: str = "bczyx"

    def __init__(self, b: int, c: int, z: int, y: int, x: int):
        super().__init__(b=b, c=c, z=z, y=y, x=x)

    def drop_batch(self) -> "Point3D":
        return Point3D(c=self.c, z=self.z, y=self.y, x=self.x)


class BatchPoint4D(BatchPointBase):
    order: str = "btczyx"

    def __init__(self, b: int, t: int, c: int, z: int, y: int, x: int):
        super().__init__(b=b, t=t, c=c, z=z, y=y, x=x)

    def drop_batch(self):
        return Point4D(t=self.t, c=self.c, z=self.z, y=self.y, x=self.x)


class PointBase(PointAndBatchPointBase):
    def __init__(self, **kwargs):
        assert all([a in self.order for a in kwargs])
        assert all([a in kwargs for a in self.order])
        super().__init__(**kwargs)

    @staticmethod
    def from_spacetime(c: int, spacetime: Sequence[int]) -> Union["Point2D", "Point3D", "Point4D"]:
        """
        :return: a suitable BatchPoint instance
        :raises: ValueError
        """
        if len(spacetime) == 4:
            t, z, y, x = spacetime
            return Point4D(t, c, z, y, x)
        elif len(spacetime) == 3:
            return Point3D(c, *spacetime)
        elif len(spacetime) == 2:
            return Point2D(c, *spacetime)
        else:
            raise ValueError(f"Uninterpretable spacetime: {spacetime}")

    def as_2d(self) -> "Point2D":
        return Point2D(c=self.c, y=self.y, x=self.x)

    def as_3d(self) -> "Point3D":
        return Point3D(c=self.c, z=self.z, y=self.y, x=self.x)

    def as_4d(self) -> "Point4D":
        return Point4D(t=self.t, c=self.c, z=self.z, y=self.y, x=self.x)

    def drop_batch(self) -> "PointBase":
        return self


class Point2D(PointBase):
    order: str = "cyx"

    def __init__(self, c: int, y: int, x: int):
        super().__init__(c=c, y=y, x=x)

    def add_batch(self, b: int) -> BatchPoint2D:
        return BatchPoint2D(b=b, c=self.c, y=self.y, x=self.x)


class Point3D(PointBase):
    order: str = "czyx"

    def __init__(self, c: int, z: int, y: int, x: int):
        super().__init__(c=c, z=z, y=y, x=x)

    def add_batch(self, b: int) -> BatchPoint3D:
        return BatchPoint3D(b=b, c=self.c, z=self.z, y=self.y, x=self.x)


class Point4D(PointBase):
    order: str = "tczyx"

    def __init__(self, t: int, c: int, z: int, y: int, x: int):
        super().__init__(t=t, c=c, z=z, y=y, x=x)

    def add_batch(self, b: int) -> BatchPoint4D:
        return BatchPoint4D(b=b, t=self.t, c=self.c, z=self.z, y=self.y, x=self.x)


class SetDeviceReturnType(NamedTuple):
    training_shape: Tuple[int, ...]
    valid_shapes: List[Tuple[int, ...]]
    shrinkage: Tuple[int, ...]
