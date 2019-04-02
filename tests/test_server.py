import pytest

from tiktorch.server import TikTorchServer
from tiktorch.rpc import Shutdown
from tiktorch.types import NDArray, NDArrayBatch

@pytest.fixture
def srv():
    return TikTorchServer()


def test_tiktorch_server_ping(srv):
    assert srv.ping() == b"pong"

def test_load_model(srv, nn_sample):
    try:
        srv.load_model(nn_sample.config, nn_sample.model, nn_sample.state, b"", [])
        print(srv.active_children())
    finally:
        shutdown_raised = False
        try:
            srv.shutdown()
        except Shutdown:
            shutdown_raised = True

        assert shutdown_raised
