"""
Types defining interop between client and server
"""
# This file should only contain types built on primitives
# available both on ilastik and tiktorch side (e.g. numpy, python stdlib)
from typing import List, Tuple, Optional, Union, Sequence

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


class PointAndBatchPointBase:
    order: str = ""
    b: int
    t: int
    c: int
    z: int
    y: int
    x: int

    def __init__(self, b: int = 0, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        self.b = b
        self.t = t
        self.c = c
        self.z = z
        self.y = y
        self.x = x
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

    def as_d(self, d: int) -> "PointAndBatchPointBase":
        """
        :param d: number of spacial dimensions
        """
        if d == 2:
            return self.as_2d()
        elif d == 3:
            return self.as_3d()
        elif d == 4:
            return self.as_4d()
        else:
            raise NotImplementedError(f"Unclear number of dimensions d={d}")

    def as_2d(self):
        raise NotImplementedError("To be implemented in subclass!")

    def as_3d(self):
        raise NotImplementedError("To be implemented in subclass!")

    def as_4d(self):
        raise NotImplementedError("To be implemented in subclass!")

    def drop_batch(self):
        raise NotImplementedError("To be implemented in subclass!")


class BatchPointBase(PointAndBatchPointBase):
    def __init__(self, b: int = 0, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(b=b, t=t, c=c, z=z, y=y, x=x)

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


class BatchPoint2D(BatchPointBase):
    order: str = "bcyx"

    def __init__(self, b: int = 0, c: int = 0, y: int = 0, x: int = 0):
        super().__init__(b=b, c=c, y=y, x=x)

    def drop_batch(self) -> "Point2D":
        return Point2D(c=self.c, y=self.y, x=self.x)


class BatchPoint3D(BatchPointBase):
    order: str = "bczyx"

    def __init__(self, b: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(b=b, c=c, z=z, y=y, x=x)

    def drop_batch(self) -> "Point3D":
        return Point3D(c=self.c, z=self.z, y=self.y, x=self.x)


class BatchPoint4D(BatchPointBase):
    order: str = "btczyx"

    def __init__(self, b: int = 0, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(b=b, t=t, c=c, z=z, y=y, x=x)

    def drop_batch(self):
        return Point4D(t=self.t, c=self.c, z=self.z, y=self.y, x=self.x)


class PointBase(PointAndBatchPointBase):
    def __init__(self, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(t=t, c=c, z=z, y=y, x=x)

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

    def __init__(self, c: int = 0, y: int = 0, x: int = 0):
        super().__init__(c=c, y=y, x=x)

    def add_batch(self, b: int = 0) -> BatchPoint2D:
        return BatchPoint2D(b=b, c=self.c, y=self.y, x=self.x)


class Point3D(PointBase):
    order: str = "czyx"

    def __init__(self, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(c=c, z=z, y=y, x=x)

    def add_batch(self, b: int = 0) -> BatchPoint3D:
        return BatchPoint3D(b=b, c=self.c, z=self.z, y=self.y, x=self.x)


class Point4D(PointBase):
    order: str = "tczyx"

    def __init__(self, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(t=t, c=c, z=z, y=y, x=x)

    def add_batch(self, b: int = 0) -> BatchPoint4D:
        return BatchPoint4D(b=b, t=self.t, c=self.c, z=self.z, y=self.y, x=self.x)
