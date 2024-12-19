from typing import Any, Callable, Dict

from tiktorch.rpc import Shutdown


class RPCInterfaceMeta(type):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        exposed = {name for name, value in namespace.items() if getattr(value, "__exposed__", False)}

        for base in bases:
            if issubclass(base, RPCInterface):
                exposed |= getattr(base, "__exposedmethods__", set())

        cls.__exposedmethods__ = frozenset(exposed)
        return cls


def exposed(method: Callable[..., Any]) -> Callable[..., Any]:
    """decorator to mark method as exposed in the public API of the class"""
    method.__exposed__ = True
    return method


class RPCInterface(metaclass=RPCInterfaceMeta):
    @exposed
    def init(self, *args, **kwargs):
        """
        Initialize server

        Server initialization postponed so the client can handle errors occurring during server initialization.
        """
        raise NotImplementedError

    @exposed
    def shutdown(self) -> Shutdown:
        raise NotImplementedError


def get_exposed_methods(obj: RPCInterface) -> Dict[str, Callable[..., Any]]:
    exposed = getattr(obj, "__exposedmethods__", None)

    if not exposed:
        raise ValueError(f"Class {obj} doesn't provide public API")

    exposed_methods = {}

    for attr_name in exposed:
        attr = getattr(obj, attr_name)
        if callable(attr):
            exposed_methods[attr_name] = attr

    return exposed_methods
