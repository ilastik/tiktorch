import pytest

import io
import logging
import time

from collections import namedtuple
from unittest import mock

import numpy as np
import torch

from tiktorch.client import TikTorchClient, LocalServerHandler, wait
from tiktorch.rpc import ISerializer, serializer_for, serialize, deserialize, Client, Server, Shutdown
from tiktorch.types import NDArrayBatch
from tiktorch.serializers import NDArrayBatchSerializer, DictSerializer
from tiktorch.rpc_interface import INeuralNetworkAPI


SrvConf = namedtuple('SrvConf', ['addr', 'port', 'meta_port'])


def client_state_request(nn_sample):
    client = TikTorchClient(tiktorch_config=nn_sample[0],
                            binary_model_file=nn_sample[1],
                            binary_model_state=nn_sample[2])

    state_dict = client.request_model_state_dict()
    file = io.BytesIO(state_dict)
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)

    logging.info("Polling")
    is_running = client.training_process_is_running()
    logging.info(f"Training process running? {is_running}")

    logging.info("Sending train data and labels...")
    train_data = [np.random.uniform(size=(1, 128, 128)).astype('float32') for _ in range(10)]
    train_labels = [np.random.randint(0, 2, size=(1, 128, 128)).astype('float32') for _ in range(10)]
    client.train(train_data, train_labels, [(idx,) for idx in range(len(train_data))])
    logging.info("Sent train data and labels and waiting for 10s...")
    time.sleep(10)

    state_dict = client.request_model_state_dict()
    file = io.BytesIO(state_dict)
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)

    client.shutdown()


@pytest.fixture
def srv_config():
    return SrvConf('127.0.0.1', 5557, 5558)


def test_local_server_start(srv_config: SrvConf, nn_sample):
    srv_handler = LocalServerHandler()
    srv_handler.start(*srv_config)
    assert srv_handler.is_running
    #time.sleep(2)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.connect(f'tcp://{srv_config.addr}:{srv_config.port}')

    client = Client(INeuralNetworkAPI, sock)
    # client = TikTorchClient(
    #     tiktorch_config=nn_sample.config,
    #     binary_model_file=nn_sample.model,
    #     binary_model_state=nn_sample.state,
    #     address=srv_config.addr,
    #     port=srv_config.port,
    #     meta_port=srv_config.meta_port,
    #     start_server=False
    # )
    #assert client.ping()
    #client.load_model({'a': 1}, b'model', b'state', b'opt_state')
    client.load_model(nn_sample.config, nn_sample.model, nn_sample.state, b'')
    time.sleep(2)
    #assert client.ping()

    #srv_handler.stop()
    assert not srv_handler.is_running


import zmq
import numpy as np
from typing import List, Any, Union, Tuple, Dict


def as_frame(obj):
    if isinstance(obj, np.ndarray):
        return zmq.Frame(data=memoryview(obj))
    elif isinstance(obj, dict):
        return zmq.utils.jsonapi.dumps(obj)
    raise ValueError('Invalid type')


def deserialize_numpy(spec: dict, frame: zmq.Frame) -> np.ndarray:
    arr = np.frombuffer(frame, dtype=spec['dtype'])
    arr.shape = spec['shape']
    return arr


def serialize_numpy(arr: np.ndarray) -> Tuple[dict, memoryview] :
    return {
        'dtype': arr.dtype.str,
        'shape': arr.shape,
    }, memoryview(arr)


def assert_np_strict_equal(arr: np.ndarray, other: np.ndarray) -> None:
    """
    Arrays equal only if shape, dtype and data are the same
    """
    assert arr.dtype == other.dtype
    np.testing.assert_array_equal(arr, other)


@pytest.mark.parametrize('dtype', [
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64
])
def test_numpy_deserialization(dtype):
    expected = np.arange(10, dtype=dtype)
    frm = as_frame(expected)

    array = deserialize_numpy({
        'dtype': dtype,
        'shape': expected.shape,
    }, frm)

    assert_np_strict_equal(expected, array)


def exposed(func):
    func._exposed = True
    return func


from typing import Any, List, Generic, Iterator, TypeVar, Mapping, Callable
T = TypeVar('T')


from zmq.utils import jsonapi



def test_serialize_NDArrayBatch():
    arr1 = np.arange(10)
    arr1.shape = (2, 5)
    arr2 = np.arange(20, 29, dtype=np.int8)
    arr2.shape = (3, 3)

    nd_batch = NDArrayBatch([arr1, arr2])
    result = list(serialize(NDArrayBatch, nd_batch))
    assert len(result) == 3

    meta = jsonapi.loads(result[0].bytes)

    assert meta == [{
        'dtype': arr1.dtype.str,
        'shape': list(arr1.shape),
    }, {
        'dtype': arr2.dtype.str,
        'shape': list(arr2.shape),
    }]


def test_deserialize_NDArrayBatch():
    array = None
    def frame_gen():
        array = np.random.random(9)
        array.shape = (3, 3)
        yield zmq.Frame(b'[{"shape": [3, 3], "dtype": "<f8"}]')
        yield zmq.Frame(memoryview(array))

    gen = frame_gen()
    batch = deserialize(NDArrayBatch, gen)
    assert len(batch) == 1


def test_deserialization_advances_iterator():
    from itertools import chain
    arr1 = np.arange(10)
    arr1.shape = (2, 5)
    arr2 = np.arange(20, 29, dtype=np.int8)
    arr2.shape = (3, 3)
    nd_batch = NDArrayBatch([arr1, arr2])

    frames_iter = serialize(NDArrayBatch, nd_batch)

    def gen_test():
        yield from frames_iter
        yield 'ok'

    gen = gen_test()

    nd_batch_deserialized = deserialize(NDArrayBatch, gen)
    assert next(gen) == 'ok'


def test_serialize_deserialize_dict():
    data = {'some': 1, 'dict': 'val'}

    serialized = serialize(dict, data)
    deserialized = deserialize(dict, serialized)

    assert data == deserialized


def test_serialize_function_args():
    def func(data: dict, frames: NDArrayBatch) -> NDArrayBatch:
        pass

    data = {'a': 1}
    arr1 = np.arange(10)
    arr1.shape = (2, 5)
    arr2 = np.arange(20, 29, dtype=np.int8)
    arr2.shape = (3, 3)
    nd_batch = NDArrayBatch([arr1, arr2])
    serialized = list(serialize_args(func, [data, nd_batch]))
    assert len(serialized) == 4



def test_exposed_methods():
    class Foo:
        attr = True

        @exposed
        def exposed_func(self):
            pass

        def not_exposed(self):
            pass

        @exposed
        def other(self):
            pass

    methods = get_exposed_methods(Foo)
    assert methods == {
        b'exposed_func': Foo.exposed_func,
        b'other': Foo.other
    }

    inst = Foo()
    methods = get_exposed_methods(inst)
    assert methods == {
        b'exposed_func': inst.exposed_func,
        b'other': inst.other
    }

def test_deserialize_return():
    def func(data: dict, frames: NDArrayBatch) -> dict:
        pass

    frame = zmq.Frame(b'{"a": 1}')
    ret = deserialize_return(func, iter([frame]))
    assert ret == {'a': 1}

def test_client():
    class API:
        @exposed
        def method(self, arg1: dict, arg2: dict) -> dict:
            raise NotImplementedError


    socket = mock.Mock(spec=zmq.Socket)
    def _recv_multipart(*args, **kwargs):
        return [zmq.Frame(b'{"c": 3}')]
    socket.recv_multipart = _recv_multipart

    client = Client(API, socket)
    a = {'a': 1}
    b = {'b': 2}
    ret = client.method(a, b)
    assert ret == {'c': 3}

def test_server():
    from threading import Thread
    class Api:
        @exposed
        def ping(self) -> dict:
            return {'id': 'pong'}

        @exposed
        def sum(self, arg1: NDArrayBatch, arg2: NDArrayBatch) -> NDArrayBatch:
            print('arg1', arg1)
            return NDArrayBatch([arg1.arrays[0] + arg2.arrays[0] * 2])

        @exposed
        def shutdown(self) -> None:
            raise Shutdown()

    context = zmq.Context()
    def _target():
        api = Api()
        socket = context.socket(zmq.PAIR)
        socket.bind(f'inproc://test')
        srv = Server(api, socket)
        logging.warning('Bound')
        srv.listen()

    t = Thread(target=_target)
    t.start()

    #context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(f'inproc://test')
    cl = Client(Api, socket)
    resp = cl.ping()
    assert resp == {'id': 'pong'}
    resp = cl.sum(NDArrayBatch([np.ones(10)]), NDArrayBatch([np.ones(10)]))
    expect = np.full(shape=(10,), fill_value=3, dtype=float)
    assert_np_strict_equal(resp.arrays[0], expect)
    resp = cl.shutdown()
    print('hi')
    t.join()
    print('done')


# class Client:
#     def method(self, arg1):
#         pass

# class Serializer:
#     def __init__(self, impl=None):
#         self._impl = impl

#     def method(self, data: TensorBatch) -> TensorBatch:
#         args = None  # Some deserialization
#         res = self._impl.method(args)
#         return serialize(args)

# class ClientImpl:
#     def method(self, arg1):
#         pass
