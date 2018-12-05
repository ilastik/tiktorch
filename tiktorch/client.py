import os
import logging
import subprocess
import yaml
import zmq
import sys
import threading as thr

import numpy as np
import torch
import torch.distributed as dist

from tiktorch.tio import TikIn
import tiktorch.utils as utils

logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    torch.multiprocessing.set_start_method('spawn', force=True)


class TikTorchClient(object):
    RANK = 0
    SIZE = 2

    _START_PROCESS = False

    def __init__(self, build_directory, address='127.0.0.1', port='29500', meta_port='29501', ilp_directory=None):
        self.build_directory = build_directory
        self.addr = address
        self.port = port
        self.meta_port = meta_port
        self.ilp_directory = ilp_directory
        # Privates
        self._args: list = None
        self._process: subprocess.Popen = None
        self._config = {}
        self._zmq_context = None
        self._zmq_socket = None
        # Locks
        self._main_lock = thr.Lock()
        # Initialize
        self.read_config()
        self.init()

    def init(self):
        logger = logging.getLogger('TikTorchClient.init')
        os.environ['MASTER_ADDR'] = self.addr
        os.environ['MASTER_PORT'] = self.port
        # Build args for the server
        self._args = [
            'python',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'server.py'),
            self.build_directory,
            '--addr', self.addr,
            '--port', self.port,
            '--meta_port', self.meta_port
        ]
        # Start server
        if self._START_PROCESS:
            logger.info("Starting Server...")
            self._process = subprocess.Popen(self._args, stdout=sys.stdout)
        # Init torch distributed
        logger.info("Initializing Process Group...")
        dist.init_process_group(backend='tcp', rank=self.RANK, world_size=self.SIZE)
        # Make server for zmq
        logger.info("Setting up ZMQ Context...")
        self._zmq_context = zmq.Context()
        logger.info("Setting up ZMQ Socket...")
        self._zmq_socket = self._zmq_context.socket(zmq.PAIR)
        logger.info("Binding to socket...")
        self._zmq_socket.bind(f'tcp://{self.addr}:{self.meta_port}')
        # Send build directory
        logger.info("Sending build directory...")
        self.meta_send({'id': 'INIT.PATHS',
                        'build_dir': self.build_directory,
                        'ilp_dir': self.ilp_directory})
        logger.info("Build directory sent.")
        return self

    def terminate(self):
        self._process.terminate()
        return self

    def kill(self):
        self._process.kill()
        return self

    def is_running(self):
        return self._process.poll() is None

    def read_config(self):
        config_file_name = os.path.join(self.build_directory, 'tiktorch_config.yml')
        if not os.path.exists(config_file_name):
            raise FileNotFoundError(f"Config file not found in "
                                    f"build_directory: {self.build_directory}.")
        with open(config_file_name, 'r') as f:
            self._config.update(yaml.load(f))
        return self

    def get(self, tag, default=None, assert_exist=False):
        if assert_exist:
            assert tag in self._config, f"Tag '{tag}' not found in configuration."
        return self._config.get(tag, default)

    def meta_send(self, info_dict):
        self._zmq_socket.send_json(info_dict)
        return self

    def meta_recv(self):
        return self._zmq_socket.recv_json()

    def batch_inputs(self, inputs):
        input_shapes = self.get('input_shape', assert_exist=True)
        assert isinstance(input_shapes, (list, tuple))
        # input_shapes can either be a list of shapes or a shape. Make sure it's the latter
        if isinstance(input_shapes[0], int):
            input_shapes = [input_shapes] * len(inputs)
        elif isinstance(input_shapes[0], (list, tuple)):
            pass
        else:
            raise TypeError(f"`input_shapes` must be a list/tuple of ints or "
                            f"lists/tuples or ints. Got list/tuple of {type(input_shapes[0])}.")
        utils.assert_(len(input_shapes) == len(inputs),
                      f"Expecting {len(inputs)} inputs, got {len(input_shapes)} input shapes.",
                      ValueError)
        batches = [input.batcher(input_shape)
                   for input, input_shape in zip(inputs, input_shapes)]
        return batches

    def parse_inputs(self, inputs):
        if isinstance(inputs, TikIn):
            inputs = [inputs]
        elif isinstance(inputs, (np.ndarray, torch.Tensor)):
            inputs = [TikIn([inputs])]
        elif isinstance(inputs, (list, tuple)):
            utils.assert_(all(isinstance(input, TikIn) for input in inputs),
                          "Inputs must all be TikIn objects.")
        else:
            raise TypeError("Inputs must be list TikIn objects.")
        return inputs

    def request_dispatch(self, mode):
        self.meta_send({'id': f'DISPATCH.{mode.upper()}'})
        response = self.meta_recv()
        return response['id'] == f'DISPATCHING.{mode.upper()}'

    def forward(self, inputs: list):
        logger = logging.getLogger('TikTorchClient.forward')
        logger.info("Waiting for lock...")
        with self._main_lock:
            # Send dispatch request and wait for confirmation
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('FORWARD')
            logger.info("Request successful.")
            # Parse inputs
            inputs = self.parse_inputs(TikIn(inputs))
            # Batch inputs
            batches = self.batch_inputs(inputs)
            logger.info("Batched inputs.")
            # Make info dict to send to server
            info = {'id': 'FORWARD.BATCHSPEC',
                    'len': len(batches),
                    'shapes': tuple(batch.shape for batch in batches)}
            logger.info("Sending BatchSpec.")
            self.meta_send(info)
            # Send batch to the server
            for batch in batches:
                logger.info("Sending batch.")
                dist.send(batch, 1)
            # Receive meta data
            logger.info("Waiting for OutSpec.")
            outspec = self.meta_recv()
            assert outspec['id'] == 'FORWARD.OUTSPEC'
            logger.info("OutSpec received.")
            output_tensor = torch.zeros(*outspec['shape'])
            # Receive it
            dist.recv(tensor=output_tensor, src=1)
            logger.info("Output received.")
        # Convert to np and done
        return output_tensor.numpy()

    def train(self, data, labels):
        logger = logging.getLogger('TikTorchClient.train')
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting Dispatch")
            assert self.request_dispatch('TRAIN')
            logger.info("Request successful.")
            # Build info dict
            info = {'id': 'TRAIN.BATCHSPEC',
                    'len': len(data),
                    'data.shapes': [tuple(_data.shape) for _data in data],
                    'labels.shapes': [tuple(_label.shape) for _label in labels]}
            logger.info("Sending BatchSpec")
            self.meta_send(info)
            # Send tensors
            logger.info("Sending data and labels...")
            for _data in data:
                _data_th = torch.from_numpy(_data)
                dist.send(_data_th, dst=1)
            for _label in labels:
                _label_th = torch.from_numpy(_label)
                dist.send(_label_th, dst=1)
            logger.info("Data and labels sent.")

    def shutdown(self):
        logger = logging.getLogger('TikTorchClient.shutdown')
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('SHUTDOWN')
            logger.info("Request successful.")

    def pause(self):
        logger = logging.getLogger('TikTorchClient.pause')
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('PAUSE')
            logger.info("Request successful.")

    def resume(self):
        logger = logging.getLogger('TikTorchClient.resume')
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('RESUME')
            logger.info("Request successful.")

    def training_process_is_running(self):
        logger = logging.getLogger("TikTorchClient.training_process_is_running")
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('POLL_TRAIN')
            # Receive info
            info = self.meta_recv()
        return info['is_alive']


def debug_client():
    TikTorchClient.read_config = lambda self: self
    TikTorchClient._START_PROCESS = False
    client = TikTorchClient(build_directory='.', address='127.0.0.1', port='29500',
                            meta_port='29501')
    client._model = torch.nn.Conv2d(1, 1, 1)
    client._config = {'input_shape': [1, 512, 512],
                      'dynamic_input_shape': '(32 * (nH + 1), 32 * (nW + 1))'}
    return client


def test_client_forward():
    BUILD_DIR = '/Users/nasimrahaman/Documents/Python/tiktorch/tests/CREMI_DUNet_pretrained'
    TikTorchClient._START_PROCESS = False
    client = TikTorchClient(BUILD_DIR)
    logging.info("Obtained client. Forwarding...")
    out = client.forward([np.random.uniform(size=(256, 256)).astype('float32') for _ in range(1)])
    logging.info(f"out.shape = {out.shape}")
    logging.info("Forwarding again...")
    out = client.forward([np.random.uniform(size=(256, 256)).astype('float32') for _ in range(1)])
    logging.info(f"out.shape = {out.shape}")
    logging.info("Shutting down...")
    client.shutdown()
    logging.info("Done!")


def test_client_train():
    BUILD_DIR = '/home/ial/Python/scratch/CREMI_DUNet_pretrained'
    ILP_DIR = '/home/ial/Python/scratch/mock_ilp_path'
    TikTorchClient._START_PROCESS = False
    client = TikTorchClient(BUILD_DIR, ilp_directory=ILP_DIR)
    logging.info("Obtained client. Forwarding...")
    out = client.forward([np.random.uniform(size=(256, 256)).astype('float32') for _ in range(1)])
    logging.info(f"out.shape = {out.shape}")

    logging.info("Polling")
    is_running = client.training_process_is_running()
    logging.info(f"Training process running? {is_running}")

    logging.info("Sending train data and labels.")
    train_data = [np.random.uniform(size=(1, 256, 256)).astype('float32') for _ in range(4)]
    train_labels = [np.random.randint(0, 2, size=(1, 256, 256)).astype('float32') for _ in range(4)]
    client.train(train_data, train_labels)
    logging.info("Sent train data and labels and waiting for 15s...")

    import time
    time.sleep(15)

    logging.info("Polling")
    is_running = client.training_process_is_running()
    logging.info(f"Training process running? {is_running}")

    logging.info("Pausing...")
    client.pause()

    logging.info("Paused training, waiting for 10s...")
    time.sleep(10)

    logging.info("Polling")
    is_running = client.training_process_is_running()
    logging.info(f"Training process running? {is_running}")

    logging.info("Forwarding again with paused model...")
    out = client.forward([np.random.uniform(size=(256, 256)).astype('float32') for _ in range(1)])
    logging.info(f"out.shape = {out.shape}")

    logging.info("Resuming training...")
    client.resume()
    logging.info("Resumed, waiting for 10s...")

    time.sleep(10)

    logging.info("Polling")
    is_running = client.training_process_is_running()
    logging.info(f"Training process running? {is_running}")

    logging.info("Forwarding again...")
    out = client.forward([np.random.uniform(size=(256, 256)).astype('float32') for _ in range(1)])
    logging.info(f"out.shape = {out.shape}")

    logging.info("Waiting 10s...")
    time.sleep(10)

    logging.info("Shutting down...")
    client.shutdown()
    logging.info("Done!")


if __name__ == '__main__':
    # import sys
    # print('Python %s on %s' % (sys.version, sys.platform))
    test_client_train()
