import pytest

from tiktorch.server import TikTorchServer
from tiktorch.rpc import Shutdown
from tiktorch.types import NDArray, NDArrayBatch


@pytest.fixture
def srv():
    tik_srv = TikTorchServer()
    yield tik_srv
    shutdown_raised = False
    try:
        tik_srv.shutdown()
    except Shutdown:
        shutdown_raised = True

    assert shutdown_raised


def test_tiktorch_server_ping(srv):
    assert srv.ping() == b"pong"


def test_load_model(srv, nn_sample):
    srv.load_model(nn_sample.config, nn_sample.model, nn_sample.state, b"", [])
    assert "Handler" in srv.active_children()


def test_forward_pass(datadir, srv, nn_sample):
    import os
    import numpy as np

    input_arr = np.load(os.path.join(datadir, "fwd_input.npy"))
    out_arr = np.load(os.path.join(datadir, "fwd_out.npy"))
    out_arr.shape = (1, 320, 320)

    srv.load_model(nn_sample.config, nn_sample.model, nn_sample.state, b"", [])

    res = srv.forward(NDArray(input_arr)).result(timeout=10)
    res_numpy = res.as_numpy()
    np.testing.assert_array_almost_equal(res_numpy[0], out_arr)
