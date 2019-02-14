import argparse
import logging
import io
import os
from importlib import util as imputils
import zmq
from zmq.utils import jsonapi

import numpy as np
import torch
import yaml
from datetime import datetime
import socket
import shutil
import tempfile

import tiktorch.utils as utils
from tiktorch.rpc import Server, Shutdown
from tiktorch.device_handler import ModelHandler
from tiktorch import serializers
from tiktorch.types import NDArray, NDArrayBatch

from .rpc_interface import INeuralNetworkAPI, IFlightControl
from typing import List, Any


if torch.cuda.is_available():
    torch.multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TikTorchServer')



from functools import wraps


class State:
    INIT = 'init'
    WORKING = 'working'

    STATES = {
        INIT: [WORKING],
        WORKING: [INIT],
    }
    def __init__(self):
        self.current = self.INIT

    def next(self, state):
        allowed = self.STATES.get(self.current)
        if state not in allowed:
            raise Exception(f'Transition from {self._current} to {state} not allowed')

    def __repr__(self):
        return f'State({self.current})'


def expect_state(*allowed):
    def decorator(func):
        return func
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            if self.state.current not in allowed:
                raise Exception(f'Call to {func} not allowed in {self.state}')

            func(*args, **kwargs)

        return wrapped
    return decorator


class TikTorchServer(INeuralNetworkAPI, IFlightControl):
    RANK = 1
    SIZE = 2

    def __init__(self, device=None):
        logger = logging.getLogger("TikTorchServer.__init__")
        # Privates
        self._build_directory = None
        self._handler: ModelHandler = None
        self._model = None
        self._log_directory = None
        # Set up queues
        if device is None:
            # The default behaviour is to select a GPU if one is availabe.
            # This can be overriden by providing device in the constructor.
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        logger.info(f"Using device: {self._device}")
        # Publics
        self.ilp_directory = None
        self.state = State()
        #self.init()
        self.tmp_dir = tempfile.mkdtemp()  # todo: get rid of tmp dir!

        self.binary_model_file = None
        self.binary_model_state = None
        self._config = None


    def __del__(self):
        shutil.rmtree(self.tmp_dir)

    def recv_init(self, config, binary_model_file, binary_model_state, binary_optimizer_state):
        print('call here')
        print(config)
        self._config = config
        logger.info("tiktorch config received.")
        self.binary_model_file = binary_model_file
        logger.info("model file received.")
        self.binary_model_state = binary_model_state
        logger.info("model state received.")
        self.binary_optimizer_state = binary_optimizer_state
        logger.info("Init data received.")
        self.load_model()
        self.state.next(State.WORKING)

    def init(self):
        logger = logging.getLogger('TikTorchServer.init')
        logger.info("Setting up ZMQ")
        # Init ZMQ
        logger.info("Setting up ZMQ Context...")
        self._zmq_context = zmq.Context()
        logger.info("Setting up ZMQ Socket...")
        self._zmq_socket = self._zmq_context.socket(zmq.PAIR)
        logger.info("Binding to socket tcp://%s:%s", self.addr, self.port)
        self._zmq_socket.bind(f'tcp://{self.addr}:{self.port}')
        # Receive build directory
        # ping = self._zmq_socket.recv_json()
        # assert ping['id'] == 'PING'
        # self._zmq_socket.send_json({'id': 'PONG'})
        #self.recv_init()
        logger.info("Waiting for init data...")

    def meta_send(self, info_dict, flags=0):
        self._zmq_socket.send_json(info_dict, flags=flags)
        return self

    def meta_recv(self):
        return self._zmq_socket.recv_json()

    def tensor_send(self, x, key):
        assert torch.is_tensor(x) or isinstance(x, np.ndarray)
        # Send meta data
        self.meta_send({'id': f"{key.upper()}.TENSORSPEC",
                        'shape': tuple(x.shape),
                        'dtype': str(x.dtype).lstrip('torch.'),
                        'device': str(x.device) if torch.is_tensor(x) else 'cpu'})
        # Make sure x is on the CPU and send
        self._zmq_socket.send((x.cpu().numpy() if torch.is_tensor(x) else x), copy=False)

    def tensor_recv(self, key, framework='numpy'):
        tensor_spec = self.meta_recv()
        assert tensor_spec['id'] == f"{key.upper()}.TENSORSPEC", (tensor_spec['id'], f"{key.upper()}.TENSORSPEC")
        # Receive the buffer
        buf = memoryview(self._zmq_socket.recv())
        x = np.frombuffer(buf, dtype=tensor_spec['dtype'].lstrip('torch.')).reshape(tensor_spec['shape'])
        if framework == 'torch':
            x = torch.from_numpy(x).to(tensor_spec['device'])
        return x

    @property
    def output_shape(self):
        return self.get('output_shape')

    @property
    def halo(self):
        """
        Returns the halo in dynamic base shape blocks
        """
        assert self.handler is not None
        halo_block = self.handler.halo_in_blocks
        base_shape = self.handler.dynamic_shape.base_shape
        return [shape*block for shape, block in zip(base_shape, halo_block)]

    @property
    def log_directory(self):
        if self._log_directory is None and self.ilp_directory is not None:
            # Make a log directory in the ilp_directory
            path = os.path.join(self.ilp_directory, 'TikTorchLogs',
                                f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}")
            os.makedirs(path, exist_ok=True)
            self._log_directory = path
            return self._log_directory
        else:
            return self._log_directory

    @property
    def model(self):
        return self.handler.model

    @property
    def device(self):
        return self.handler.device

    @property
    def handler(self):
        assert self._handler
        return self._handler

    def dry_run(self, image_shape, train=False):
        """
        Initiates dry run.
        Parameters
        ----------
        image_shape: list or tuple
        shape of an image in the dataset (e.g `HW` for 2D or `DHW` for 3D)
        """
        assert self.handler is not None
        return self.handler.binary_dry_run(list(image_shape), train_flag=train)

    def _set_handler(self, model):
        assert self.get('input_shape') is not None
        # Pass
        self._handler = ModelHandler(model=model,
                                     device_names=self._device,
                                     channels=self.get('input_shape')[0],
                                     dynamic_shape_code=self.get('dynamic_input_shape'),
                                     log_directory=self.log_directory)

    def get(self, tag, default=None, assert_exist=False):
        if assert_exist:
            assert tag in self._config, f"Tag '{tag}' not found in configuration."
        return self._config.get(tag, default)

    @staticmethod
    def define_patched_model(model_file_name, model_class_name, model_init_kwargs):
        # Dynamically import file.
        module_spec = imputils.spec_from_file_location('model', model_file_name)
        module = imputils.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        # Build model from file
        model: torch.nn.Module = \
            getattr(module, model_class_name)(**model_init_kwargs)
        # Monkey patch
        model.__model_file_name = model_file_name
        model.__model_class_name = model_class_name
        model.__model_init_kwargs = model_init_kwargs
        return model

    def load_model(
        self,
        config: dict,
        model_file: bytes,
        model_state: bytes,
        optimizer_state: bytes,
    ) -> None:
        # Dynamically import file.
        # hack to keep current impl: safe binary_model_file to tmp file
        # todo: load model.py without tmp directory
        self.binary_model_file = model_file
        self.binary_model_state = model_state
        self._config = config

        tmp_dir = self.tmp_dir
        model_file_path = os.path.join(tmp_dir, 'model.py')
        with open(model_file_path, 'wb') as f:
            f.write(self.binary_model_file)

        model_state_path = None
        if self.binary_model_state:
            model_state_path = os.path.join(tmp_dir, 'state.nn')
            with open(model_state_path, 'wb') as f:
                f.write(self.binary_model_state)

        # todo: optimizer state

        model = utils.define_patched_model(model_file_path,
                                           self.get('model_class_name'),
                                           self.get('model_init_kwargs'))
        # Load parameters
        try:
            state_dict = torch.load(model_state_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except:
            logger.warning(f"state.nn file not found in {model_state_path}, not loading weights!")
            # raise FileNotFoundError(f"Model weights could not be found at location '{state_path}'!")
        # Build handler
        self._set_handler(model)

    def forward(self, batch: NDArrayBatch) -> NDArrayBatch:
        # TODO: Use TikIO for batching
        tensors = [torch.from_numpy(a) for a in batch.as_numpy()]
        res = self.handler.forward(*tensors)
        logger.debug("Send forward result")
        return NDArrayBatch([NDArray(res.numpy())])

    def train(self, data: NDArrayBatch, labels: NDArrayBatch) -> None:
        torch_data = [
            torch.from_numpy(arr) for arr in data.as_numpy()
        ]
        torch_labels = [
            torch.from_numpy(arr) for arr in labels.as_numpy()
        ]
        logger.info("Received data and labels from chief.")
        logger.info("Sending to handler.")
        self.handler.train(torch_data, torch_labels)
        logger.info("Sent to handler.")

    def training_process_is_running(self) -> bool:
        return self.handler.training_process_is_alive()

    def pause(self) -> None:
        self.handler.pause_training()

    def resume(self) -> None:
        self.handler.resume_training()

    def set_hparams(self, params: dict) -> None:
        logger.info("Sending to handler.")
        self.handler.set_hparams(params)
        logger.info("Sent to handler.")

    def listen(self):
        logger = logging.getLogger('TikTorchServer.listen')
        logger.info('Waiting...')
        # Listen for requests
        while True:
            request, args = self._zmq_socket.recv_serialized(deserialize, copy=False)
            #request = self.meta_recv()
            logger.info("Request Received.")
            if request['id'] == 'INIT':
                self.recv_init(*args)
            elif request['id'] == 'PING':
                self.meta_send({'id': 'PONG'})
            elif request['id'] == 'DISPATCH.FORWARD':
                logger.info("Received request to dispatch forward.")
                # Confirm dispatch
                self.meta_send({'id': 'DISPATCHING.FORWARD'})
                logger.info("Dispatch confirmed.")
                self.forward()
                logger.info("Forward successful; waiting...")
            elif request['id'] == 'DISPATCH.TRAIN':
                logger.info("Received request to dispatch train.")
                # Confirm dispatch
                self.meta_send({'id': 'DISPATCHING.TRAIN'})
                logger.info('Dispatch confirmed.')
                self.train()
                logger.info("Train successful; waiting...")
            elif request['id'] == 'DISPATCH.SHUTDOWN':
                logger.info("Received request to shutdown.")
                self.meta_send({'id': 'DISPATCHING.SHUTDOWN'})
                logger.info("Dispatch confirmed.")
                self.shutdown()
                break
            elif request['id'] == 'DISPATCH.PAUSE':
                logger.info("Received request to pause training.")
                self.meta_send({'id': 'DISPATCHING.PAUSE'})
                logger.info("Dispatch confirmed, pausing training...")
                self.handler.pause_training()
            elif request['id'] == 'DISPATCH.RESUME':
                logger.info("Received request to resume training.")
                self.meta_send({'id': 'DISPATCHING.RESUME'})
                logger.info("Dispatch confirmed, resuming...")
                self.handler.resume_training()
            elif request['id'] == 'DISPATCH.POLL_TRAIN':
                logger.info("Received request to poll training process.")
                self.meta_send({'id': 'DISPATCHING.POLL_TRAIN'})
                logger.info("Dispatch confirmed, polling...")
                self.poll_training_process()
            elif request['id'] == 'DISPATCH.HYPERPARAMETERS':
                logger.info("Received request to dispatch hyperparameters.")
                self.meta_send({'id': 'DISPATCHING.HYPERPARAMETERS'})
                logger.info("Dispatch confirmed, changing hyperparameters...")
                self.set_hparams()
            elif request['id'] == 'DISPATCH.MODEL_STATE_DICT_REQUEST':
                logger.info("Received a request for the model state dict.")
                self.meta_send({'id': 'DISPATCHING.MODEL_STATE_DICT_REQUEST'})
                logger.info('Dispatch confirmed.')
                self.request_model_state_dict()
            else:
                # Bad id
                raise RuntimeError(request)

    def request_model_state_dict(self):
        logger = logging.getLogger('TikTorchServer.request_model_state_dict')
        logger.info('Requesting model state dict from handler....')
        self.handler.update_state()
        state_dict = io.BytesIO()
        torch.save(self.model.state_dict(), f=state_dict)
        logger.info('Sending state dict.')
        self._zmq_socket.send(state_dict.getvalue())
        logger.info('Sent state dict.')
        state_dict.close()


    def request_optimizer_state_dict(self):
        pass


    def poll_training_process(self):
        logger = logging.getLogger('TikTorchServer.poll_training_process')
        logger.info("Polling...")
        # Check if training process is running, and send info back
        it_lives = self.handler.training_process_is_alive()
        logger.info("Poll successful. Sending response...")
        info = {'id': 'POLL_TRAIN.INFO',
                'is_alive': it_lives}
        self.meta_send(info)
        logger.info("Poll response sent.")

    def ping(self) -> bytes:
        return b'pong'

    def shutdown(self):
        logger = logging.getLogger('TikTorchServer.shutdown')
        logger.info("Stopping training...")
        if self._handler:
            self._handler.stop_training()
        logger.info("Training has stopped")
        raise Shutdown()


def debug_server():
    TikTorchServer.read_config = lambda self: self
    server = TikTorchServer(address='127.0.0.1', port='29500',
                            meta_port='29501')
    server._model = torch.nn.Conv2d(1, 1, 1)
    server._config = {'input_shape': [1, 512, 512],
                      'dynamic_input_shape': '(32 * (nH + 1), 32 * (nW + 1))'}
    server._set_handler(server._model)
    return server


class ServerProcess:
    def __init__(self, address='127.0.0.1', port='29500', device=None, **kwargs):
        # TODO: Remove metaport
        self._addr = address
        self._port = port
        self._device = device

    def listen(self):
        api_provider = TikTorchServer(device=self._device)
        srv = Server(api_provider, f'tcp://{self._addr}:{self._port}')
        srv.listen()


if __name__ == '__main__':
    # Output pid for process tracking
    print(os.getpid(), flush=True)

    parsey = argparse.ArgumentParser()
    parsey.add_argument('--addr', type=str, default='127.0.0.1')
    parsey.add_argument('--port', type=str, default='29500')
    parsey.add_argument('--meta_port', type=str, default='29501')
    parsey.add_argument('--debug', type=bool, default=False)
    args = parsey.parse_args()

    srv = ServerProcess(
        address=args.addr,
        port=args.port,
    )
    srv.listen()
