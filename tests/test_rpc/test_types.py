from concurrent.futures import CancelledError, Future
from typing import Any

import pytest

from tiktorch.rpc import RPCFuture
from tiktorch.rpc.types import isfutureret


class TExc(Exception):
    """Intentional exception"""

    pass


def process_err(val):
    raise TExc()


def test_future_map_returns_new_future():
    f = RPCFuture()
    new = f.map(lambda v: v + 2)
    f.set_result(40)
    assert new.result() == 42


def test_future_map_raise_exception():
    f = RPCFuture()
    new = f.map(process_err)
    f.set_result(40)

    with pytest.raises(TExc):
        assert new.result()


def test_future_on_chained_exception():
    f = RPCFuture()
    new = f.map(lambda v: v + 2)
    f.set_exception(TExc())

    with pytest.raises(TExc):
        assert new.result()


def test_rpcfuture_attach():
    rpc_fut = RPCFuture()
    fut = Future()

    rpc_fut.attach(fut)

    fut.set_result(42)

    assert rpc_fut.result(timeout=1) == 42


def test_rpcfuture_cancellation():
    rpc_fut = RPCFuture()
    fut = Future()

    rpc_fut.attach(fut)

    rpc_fut.cancel()

    assert fut.cancelled()


def test_propagates_only_once_1():
    rpc_fut = RPCFuture()
    fut = Future()

    rpc_fut.attach(fut)
    rpc_fut.cancel()
    rpc_fut.set_result(42)

    with pytest.raises(CancelledError):
        assert fut.result(timeout=1)


def test_propagates_only_once_2():
    rpc_fut = RPCFuture()
    fut = Future()

    rpc_fut.attach(fut)
    fut.set_result(42)
    fut.cancel()

    assert rpc_fut.result(timeout=1) == 42


def case_object() -> object():
    return object()


def case_elipsis() -> ...:
    return ...


def case_none() -> None:
    return ...


def case_not_annotated():
    pass


def case_any() -> Any:
    pass


@pytest.mark.parametrize("func", [case_object, case_elipsis, case_none, case_not_annotated, case_any])
def test_isfutureret_doesnt_raise_on_object_returns(func):
    assert not isfutureret(func)


def case_rpc_future() -> RPCFuture:
    return RPCFuture()


def case_future() -> Future:
    return Future()


def case_rpc_future_generic() -> RPCFuture[int]:
    return RPCFuture()


@pytest.mark.parametrize("func", [case_rpc_future, case_future, case_rpc_future_generic])
def test_isfutureret_generic(func):
    assert isfutureret(func)
