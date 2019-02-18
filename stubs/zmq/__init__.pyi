from typing import Optional, Union, Iterable

bytes_t = bytes


class Frame:
    def __init__(self, data: Optional[Union[memoryview, bytes]] = None) -> None:
        pass
    bytes: bytes_t
    buffer: memoryview


class Socket:
    def __init__(self, *args, **kwargs) -> None: ...

    def send_multipart(
        self,
        msg_parts: Iterable[Union[memoryview, bytes, Frame]],
        copy: Optional[bool] = None
    ): ...

    def recv_multipart(
        self,
        flags: int = 0,
        copy: bool = True,
        track: bool = False,
    ): ...

    def bind(self, addr: str) -> None: ...

    def setsockopt(self, option: int, val: Union[int,bytes]) -> None: ...


class Context:
    def socket(self, socket_type: int, **kwargs) -> Socket: ...

    @classmethod
    def instance(cls) -> 'Context': ...


class Poller:
    pass

POLLIN: int
POLLOUT: int
REP: int
REQ: int
LINGER: int
