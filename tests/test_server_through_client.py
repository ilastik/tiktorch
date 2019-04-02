from threading import Thread

import zmq
import pytest
import numpy as np

from tiktorch.server import TikTorchServer
from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl
from tiktorch.rpc import Client, Server, RPCInterface, InprocConnConf, Shutdown
from tiktorch.types import NDArray, NDArrayBatch


@pytest.fixture
def ctx():
    return zmq.Context()


@pytest.fixture
def conn_conf(ctx):
    return InprocConnConf("test", "test_pub", ctx)


@pytest.fixture
def srv(conn_conf, client_control):
    api_provider = TikTorchServer()

    srv = Server(api_provider, conn_conf)

    def _target():
        srv.listen()

    t = Thread(target=_target)
    t.start()

    yield api_provider

    with pytest.raises(Shutdown):
        client_control.shutdown()
    t.join(timeout=2)

    assert not t.is_alive()


@pytest.fixture
def client_control(conn_conf):
    cl_fl = Client(IFlightControl(), conn_conf)

    yield cl_fl


@pytest.fixture
def client(conn_conf):
    cl = Client(INeuralNetworkAPI(), conn_conf)

    yield cl


def test_tiktorch_server_ping(srv, client_control):
    assert client_control.ping() == b"pong"


def test_load_model(srv, client, nn_sample):
    client.load_model(nn_sample.config, nn_sample.model, nn_sample.state, b"", [])
    print(client.active_children())


def test_forward_pass(datadir, srv, client, nn_sample):
    import os
    import numpy as np

    input_arr = np.load(os.path.join(datadir, "fwd_input.npy"))
    out_arr = np.load(os.path.join(datadir, "fwd_out.npy"))
    out_arr.shape = (1, 1, 320, 320)  # TODO Figure out shape difference (1, 320, 320) vs (1, 1, 320, 320)

    client.load_model(nn_sample.config, nn_sample.model, nn_sample.state, b"", [])

    res = client.forward(NDArrayBatch([NDArray(input_arr)]))
    res_numpy = res.as_numpy()
    np.testing.assert_array_almost_equal(res_numpy[0], out_arr)


def test_client_dry_run(srv, client, nn_sample):
    client.load_model(nn_sample.config, nn_sample.model, nn_sample.state, b"", [])

    client.dry_run({"train": True, "upper_bound": [1, 1, 125, 1250, 2040]})

    valid_shape = client.dry_run({"train": False, "upper_bound": [1, 1, 125, 1250, 2040]})
    assert isinstance(valid_shape, dict)
    assert "shape" in valid_shape
