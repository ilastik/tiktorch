from typing import Any, Callable, Dict


class RPCInterfaceMeta(type):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        exposed = {
            name
            for name, value in namespace.items()
            if getattr(value, "__exposed__", False)
        }

        for base in bases:
            if issubclass(base, RPCInterface):
                 exposed ^= getattr(base, "__exposedmethods__", set())

        cls.__exposedmethods__ = frozenset(exposed)
        return cls


class RPCInterface(metaclass=RPCInterfaceMeta):
    pass


def exposed(method: Callable[...,Any]) -> Callable[...,Any]:
    method.__exposed__ = True
    return method


def get_exposed_methods(obj: RPCInterface) -> Dict[str, Callable[...,Any]]:
    exposed = getattr(obj, '__exposedmethods__', None)

    if not exposed:
        raise ValueError(f"Class doesn't provide public API")

    exposed_methods = {}

    for attr_name in exposed:
        attr = getattr(obj, attr_name)
        if callable(attr):
            exposed_methods[attr_name] = attr

    return exposed_methods
