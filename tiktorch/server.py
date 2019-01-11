import logging
import os
from importlib import util as imputils
import zmq

import numpy as np
import torch
import yaml
from datetime import datetime
import socket

import tiktorch.utils as utils
from tiktorch.device_handler import ModelHandler


if torch.cuda.is_available():
    torch.multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TikTorchServer')


class TikTorchServer(object):
    RANK = 1
    SIZE = 2

    def __init__(self, address='127.0.0.1', port='29500',
                 meta_port='29501', device=None):
        logger = logging.getLogger("TikTorchServer.__init__")
        # Privates
        self._build_directory = None
        self._handler: ModelHandler = None
        self._model = None
        self._config = {}
        self._log_directory = None
        # Set up queues
        self._zmq_context: zmq.Context = None
        self._zmq_socket: zmq.Socket = None
        self._zmq_pollin: zmq.Poller = None
        if device is None:
            # The default behaviour is to select a GPU if one is availabe.
            # This can be overriden by providing device in the constructor.
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        logger.info(f"Using device: {self._device}")
        # Publics
        self.ilp_directory = None
        self.addr = address
        self.port = port
        self.meta_port = meta_port
        self.init()
        self.read_config()

    def init(self):
        logger = logging.getLogger('TikTorchServer.init')
        logger.info("Setting up ZMQ")
        # Init ZMQ
        logger.info("Setting up ZMQ Context...")
        self._zmq_context = zmq.Context()
        logger.info("Setting up ZMQ Socket...")
        self._zmq_socket = self._zmq_context.socket(zmq.PAIR)
        logger.info("Binding to socket...")
        self._zmq_socket.connect(f'tcp://{self.addr}:{self.port}')
        logger.info("Setting up Poller...")
        self._zmq_pollin = zmq.Poller()
        self._zmq_pollin.register(self._zmq_socket, zmq.POLLIN)
        # Receive build directory
        logger.info("Waiting for build directory...")
        message = self.meta_recv()
        assert message['id'] == 'INIT.PATHS'
        self.build_directory = message['build_dir']
        self.ilp_directory = message['ilp_dir']
        logger.info("Build directory received.")

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
    def build_directory(self):
        if self._build_directory is not None:
            return self._build_directory
        else:
            raise ValueError("Trying to access `build_directory`, but it's not set yet.")

    @build_directory.setter
    def build_directory(self, value):
        if not os.path.exists(value):
            raise FileNotFoundError(f"Build directory does not exist: {value}")
        self._build_directory = value

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
        if self._handler is None:
            self.load_model()
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

    def read_config(self):
        config_file_name = os.path.join(self.build_directory, 'tiktorch_config.yml')
        if not os.path.exists(config_file_name):
            raise FileNotFoundError(f"Config file not found in "
                                    f"build_directory: {self.build_directory}.")
        with open(config_file_name, 'r') as f:
            self._config.update(yaml.load(f))
        return self

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

    def load_model(self):
        # Dynamically import file.
        model_file_name = os.path.join(self.build_directory, 'model.py')
        model = utils.define_patched_model(model_file_name,
                                           self.get('model_class_name'),
                                           self.get('model_init_kwargs'))
        # model = DUNet(**self.get('model_init_kwargs'))
        # Load parameters
        state_path = os.path.join(self.build_directory, 'state.nn')
        try:
            state_dict = torch.load(state_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except:
            logger.warning(f"state.nn file not found in {state_path}, not loading weights!")
            # raise FileNotFoundError(f"Model weights could not be found at location '{state_path}'!")
        # Build handler
        self._set_handler(model)
        return self

    def forward(self):
        logger = logging.getLogger('TikTorchServer.forward')
        logger.info("Receiving BatchSpec...")
        batch_spec = self.meta_recv()
        assert batch_spec['id'] == 'FORWARD.BATCHSPEC'
        logger.info("Received BatchSpec.")
        batches = [torch.zeros(*shape) for shape in batch_spec['shapes']]
        for idx in range(batch_spec['len']):
            logger.info("Receiving batch from chief.")
            batches[idx] = self.tensor_recv(f'FORWARD_IN_{idx}', framework='torch')
        # Forward
        logger.info("Feedforward.")
        output_batches = self.handler.forward(*batches)
        # Send output spec
        # logger.info("Sending OutSpec.")
        # self.meta_send({'id': 'FORWARD.OUTSPEC', 'shape': tuple(output_batches.shape)})
        logger.info("Sending output.")
        self.tensor_send(output_batches, 'FORWARD_OUT')
        logger.info("Sent output.")

    def train(self):
        logger = logging.getLogger('TikTorchServer.train')
        logger.info("Receiving BatchSpec")
        batch_spec = self.meta_recv()
        assert batch_spec['id'] == 'TRAIN.BATCHSPEC'
        assert batch_spec['len'] == len(batch_spec['data.shapes']) == len(batch_spec['labels.shapes']) == \
               len(batch_spec['sample_ids'])
        logger.info("Receiving data and labels from chief.")
        data = [torch.zeros(*shape) for shape in batch_spec['data.shapes']]
        labels = [torch.zeros(*shape) for shape in batch_spec['labels.shapes']]
        ids = [None] * batch_spec['len']
        # Receive tensors
        for idx, id in enumerate(batch_spec['sample_ids']):
            data[idx] = self.tensor_recv(f'TRAIN_DATA_{id}')
            labels[idx] = self.tensor_recv(f'TRAIN_LABEL_{id}')
        logger.info("Received data and labels from chief.")
        logger.info("Sending to handler.")
        self.handler.train(data, labels)
        logger.info("Sent to handler.")

    def set_hparams(self):
        logger = logging.getLogger('TikTorchServer.set_hparams')
        logger.info("Receiving new hyperparameters from client.")
        hparams = self.meta_recv()
        assert hparams['id'] == 'TRAIN.HYPERPARAMETERS'
        logger.info("Sending to handler.")
        self.handler.set_hparams(hparams['parameters'])
        logger.info("Sent to handler.")

    def listen(self):
        logger = logging.getLogger('TikTorchServer.listen')
        logger.info('Waiting...')
        # Listen for requests
        while True:
            socks = dict(self._zmq_pollin.poll(50))
            if socks:
                # Yay, a message!
                if socks.get(self._zmq_socket) == zmq.POLLIN:
                    logger.info("Request Polled.")
                    request = self.meta_recv()
                    logger.info("Request Received.")
                    if request['id'] == 'DISPATCH.FORWARD':
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
                    else:
                        # Bad id
                        raise RuntimeError

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

    def shutdown(self):
        logger = logging.getLogger('TikTorchServer.shutdown')
        logger.info("Stopping training...")
        self.handler.stop_training()
        logger.info("Training stop.")


def debug_server():
    TikTorchServer.read_config = lambda self: self
    server = TikTorchServer(address='127.0.0.1', port='29500',
                            meta_port='29501')
    server._model = torch.nn.Conv2d(1, 1, 1)
    server._config = {'input_shape': [1, 512, 512],
                      'dynamic_input_shape': '(32 * (nH + 1), 32 * (nW + 1))'}
    server._set_handler(server._model)
    return server


if __name__ == '__main__':
    import argparse
    parsey = argparse.ArgumentParser()
    parsey.add_argument('--addr', type=str, default='127.0.0.1')
    parsey.add_argument('--port', type=str, default='29500')
    parsey.add_argument('--meta_port', type=str, default='29501')
    parsey.add_argument('--debug', type=bool, default=False)
    args = parsey.parse_args()

    # Go!
    if args.debug:
        server = debug_server()
    else:
        server = TikTorchServer(address=args.addr, port=args.port, meta_port=args.meta_port)
    server.listen()

