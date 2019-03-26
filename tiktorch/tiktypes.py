"""
Types defining interop between processes on the server
"""
import torch

from typing import List, Tuple, Optional, Union, Sequence


class TikTensor:
    """
    Containter for pytorch tensor to transfer additional properties
    e.g. position of array in dataset (id_)
    """

    def __init__(self, tensor: torch.Tensor, id_: Optional[Tuple[int, ...]] = None, label: Optional[torch.Tensor] = None) -> None:
        self._torch = tensor
        self.id = id_
        self.label = label

    def as_torch(self, with_label=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if with_label:
            return self._torch, self.label
        else:
            return self._torch

    @property
    def dtype(self):
        return self._torch.dtype

    @property
    def shape(self):
        return self._torch.shape


class TikTensorBatch:
    """
    Batch of TikTensor
    """

    def __init__(self, tensors: List[TikTensor]):
        assert all([isinstance(t, TikTensor) for t in tensors])
        self._tensors = tensors

    def tensor_metas(self):
        return [{"dtype": t.dtype.str, "shape": t.shape, "id": t.id} for t in self._tensors]

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        for item in self._tensors:
            yield item

    def as_torch(self, with_label=False) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        return [t.as_torch(with_label=with_label) for t in self._tensors]

    @property
    def ids(self) -> List[Tuple[int]]:
        return [t.id for t in self._tensors]


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
        me, other = self.upcast_dim(self, other)
        return all([m == o for m, o in zip(me, other)])

    def __le__(self, other):
        me, other = self.upcast_dim(self, other)
        return all([m <= o for m, o in zip(me, other)])

    def __ge__(self, other):
        me, other = self.upcast_dim(self, other)
        return all([m >= o for m, o in zip(me, other)])

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
    def from_spacetime(cls, c: int, spacetime: Sequence[int]) -> Union["Point2D", "Point3D", "Point4D"]:
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

    def add_batch(self) -> BatchPoint2D:
        return BatchPoint2D(c=self.c, y=self.y, x=self.x)


class Point3D(PointBase):
    order: str = "czyx"

    def __init__(self, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(c=c, z=z, y=y, x=x)

    def add_batch(self) -> BatchPoint3D:
        return BatchPoint3D(c=self.c, z=self.z, y=self.y, x=self.x)


class Point4D(PointBase):
    order: str = "tczyx"

    def __init__(self, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(t=t, c=c, z=z, y=y, x=x)

    def add_batch(self) -> BatchPoint4D:
        return BatchPoint4D(t=self.t, c=self.c, z=self.z, y=self.y, x=self.x)