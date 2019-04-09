import pytest

from tiktorch.rpc import RPCFuture


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
