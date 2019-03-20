"""
Types defining interop between processes on the server
"""
import torch

from typing import List, Tuple, Optional, Union, TypeVar


class TikTensor:
    """
    Containter for pytorch tensor to transfer additional properties
    e.g. position of array in dataset (id_)
    """

    def __init__(self, tensor: torch.Tensor, id_: Optional[Tuple[int, ...]] = None) -> None:
        self._torch = tensor
        self.id = id_

    def as_torch(self) -> torch.Tensor:
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
        self._tensors = tensors

    def tensor_metas(self):
        return [{"dtype": t.dtype.str, "shape": t.shape, "id": t.id} for t in self._tensors]

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        for item in self._tensors:
            yield item

    def as_torch(self) -> List[torch.Tensor]:
        return torch.stack([t.as_torch() for t in self._tensors])


class BatchPointBase:
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
        return ", ".join([f"{a}:{getattr(self, a)}" for a in self.order])

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        for a in self.order:
            yield getattr(self, a)

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


class BatchPoint3D(BatchPointBase):
    order: str = "bczyx"

    def __init__(self, b: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(b=b, c=c, z=z, y=y, x=x)


class BatchPoint4D(BatchPointBase):
    order: str = "btczyx"

    def __init__(self, b: int = 0, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(b=b, t=t, c=c, z=z, y=y, x=x)


class PointBase:
    order: str = ""
    t: int
    c: int
    z: int
    y: int
    x: int

    def __init__(self, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
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
        return ", ".join([f"{a}:{getattr(self, a)}" for a in self.order])

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        for a in self.order:
            yield getattr(self, a)

    def as_2d(self) -> "Point2D":
        return Point2D(c=self.c, y=self.y, x=self.x)

    def as_3d(self) -> "Point3D":
        return Point3D(c=self.c, z=self.z, y=self.y, x=self.x)

    def as_4d(self) -> "Point4D":
        return Point4D(t=self.t, c=self.c, z=self.z, y=self.y, x=self.x)


class Point2D(PointBase):
    order: str = "cyx"

    def __init__(self, c: int = 0, y: int = 0, x: int = 0):
        super().__init__(c=c, y=y, x=x)


class Point3D(PointBase):
    order: str = "czyx"

    def __init__(self, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(c=c, z=z, y=y, x=x)


class Point4D(PointBase):
    order: str = "tczyx"

    def __init__(self, t: int = 0, c: int = 0, z: int = 0, y: int = 0, x: int = 0):
        super().__init__(t=t, c=c, z=z, y=y, x=x)
