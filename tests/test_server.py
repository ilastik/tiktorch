from threading import Thread

import zmq
import pytest
import numpy as np

from tiktorch.server import TikTorchServer
from tiktorch.rpc_interface import INeuralNetworkAPI
from tiktorch.rpc import Client, Server
from tiktorch.types import NDArray, NDArrayBatch


SOCKET_ADDR = 'inproc://test'


@pytest.fixture
def ctx():
    return zmq.Context()


@pytest.fixture
def srv(ctx):
    api_provider = TikTorchServer(device='cpu')

    socket = ctx.socket(zmq.PAIR)
    socket.bind(SOCKET_ADDR)

    srv = Server(api_provider, socket)

    def _target():
        srv.listen()

    t = Thread(target=_target)
    t.start()
    yield api_provider
    t.join(timeout=2)
    assert not t.is_alive()


@pytest.fixture
def client(ctx):
    sock = ctx.socket(zmq.PAIR)
    sock.connect(SOCKET_ADDR)
    cl = Client(INeuralNetworkAPI(), sock)
    yield cl

    cl.shutdown()


def test_tiktorch_server_ping(srv, client):
    assert client.ping() == b'pong'


def test_load_model(srv, client, nn_sample):
    assert srv.binary_model_file is None
    assert srv.binary_model_state is None
    assert srv._config is None

    client.load_model(nn_sample.config, nn_sample.model, nn_sample.state, b'')

    assert srv.binary_model_file == nn_sample.model
    assert srv.binary_model_state == nn_sample.state
    assert srv._config == nn_sample.config


def test_forward_pass(datadir, srv, client, nn_sample):
    import os
    import numpy as np
    input_arr = np.load(os.path.join(datadir, 'fwd_input.npy'))
    out_arr = np.load(os.path.join(datadir, 'fwd_out.npy'))
    out_arr.shape = (1, 1, 320, 320) # TODO Figure out shape difference (1, 320, 320) vs (1, 1, 320, 320)

    client.load_model(nn_sample.config, nn_sample.model, nn_sample.state, b'')
    res = client.forward(NDArrayBatch([NDArray(input_arr)]))
    res_numpy = res.as_numpy()
    np.testing.assert_array_almost_equal(res_numpy[0], out_arr)
