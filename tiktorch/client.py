import sys
import time

import logging
import numpy as np
import os
import subprocess
import threading as thr
import torch
import yaml
import zmq
import pickle
import warnings

from zmq.utils import jsonapi
from paramiko import SSHClient, AutoAddPolicy
from socket import gethostbyname, timeout

from typing import Sequence

from tiktorch import serializers
import tiktorch.utils as utils
from tiktorch.tio import TikIn

logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    torch.multiprocessing.set_start_method('spawn', force=True)


class TikTorchClient(object):
    RANK = 0
    SIZE = 2

    _START_PROCESS = False

    def __init__(self, tiktorch_config, binary_model_file, binary_model_state=b'', binary_optimizer_state=b'',
                 address='localhost', ssh_port=22, port='5556', meta_port='5557', start_server=True, username=None,
                 password=None):
        warnings.warn("Deprecated class, please use: Client(INeuralNetworkAPI(), ...)")
        # todo: actually use meta_port
        logger = logging.getLogger()
        self.ssh_connect = {
            'hostname': address,
            'port': ssh_port,
            'username': username,
            'password': password,
            'timeout': 10
        }
        self.ssh_max_buffer_size = 4096
        self.addr = gethostbyname(address)  # resolve address if address is a hostname
        self.port = port
        self.meta_port = meta_port
        # Privates
        self._args: list = None
        self._process: subprocess.Popen = None
        self._config = tiktorch_config
        self._zmq_context = None
        self._zmq_socket = None
        # Locks
        self._main_lock = thr.Lock()
        self._zmq_lock = thr.Lock()
        # Initialize
        self._local_server_process = None
        self._ssh_client = None
        self._remote_server_channel = None
        if start_server:
            self.start_server()

        self.init(tiktorch_config, binary_model_file, binary_model_state, binary_optimizer_state)

    def start_server(self):
        with self._main_lock:
            if self.addr in ('127.0.0.1', 'localhost'):
                logger = logging.getLogger('TikTorchClient.local_server_process')
                # start local server process
                assert self._local_server_process is None or self._local_server_process.poll() is not None, \
                    'local server already running!'
                logger.info('Starting local TikTorchServer...')
                self._local_server_process = subprocess.Popen(
                    [sys.executable, '-c',
                     f'from tiktorch.server import TikTorchServer;TikTorchServer(address="{self.addr}", '
                     f'port={self.port}, meta_port={self.meta_port}).listen()'],
                    stdout=sys.stdout)
                # check if local server process runs
                time.sleep(5)
                if self._local_server_process.poll() is not None:
                    logger.error('Could not start local TikTorchServer')
            else:
                logger = logging.getLogger('TikTorchClient.ssh_client')
                # todo: improve starting remote server
                self.kill_remote_server()  # make sure we do not have an open server connection
                assert self._ssh_client is None
                assert self._remote_server_channel is None
                self._ssh_client = SSHClient()
                self._ssh_client.set_missing_host_key_policy(AutoAddPolicy())
                self._ssh_client.load_system_host_keys()
                try:
                    self._ssh_client.connect(**self.ssh_connect)
                except timeout as e:
                    logger.error(f'Could not establish ssh connection:\n {e}')
                    return

                channel = self._ssh_client.invoke_shell()
                channel.settimeout(10)
                try:
                    channel.send('source .bashrc\n')
                    time.sleep(1)
                    channel.send('conda activate ilastikTiktorchServer\n')
                    time.sleep(1)
                    logger.info('Starting remote TikTorchServer...')
                    channel.send(
                        f'python -c \'from tiktorch.server import TikTorchServer;TikTorchServer(address="{self.addr}", '
                        f'port={self.port}, meta_port={self.meta_port}).listen()\'\n'
                    )
                except timeout as e:
                    logger.error(f'Tiktorch server could not be started:\n {e}')
                    self.kill_remote_server()
                    return

                self._remote_server_channel = channel
                self.log_server_report()

    def ping(self):
        self._zmq_socket.send_json({'id': 'PING'})
        resp = self._zmq_socket.recv_json()
        return resp['id'] == 'PONG'

    def log_server_report(self, delay: int=2):
        if self._local_server_process is not None:
            logger = logging.getLogger('TikTorchClient.local_server_process')
            logger.info(f'local server process active: {self._local_server_process.poll() is None}')
        elif self._ssh_client is not None:
            logger = logging.getLogger('TikTorchClient.ssh_client')
            for _ in range(delay):
                time.sleep(1)
                if self._remote_server_channel.recv_ready():
                    logger.info(f'{self._remote_server_channel.recv(self.ssh_max_buffer_size).decode("utf-8")}')
        else:
            logger = logging.getLogger()
            logger.warning('No local or remote server to report from!')



    def kill_server(self, delay: int=0):
        assert self._local_server_process is None or self._ssh_client is None, 'Use either a local or a remote server!'
        if self._local_server_process is not None:
            self.kill_local_server(delay=delay)
        elif self._ssh_client is not None:
            self.kill_remote_server()

    def kill_remote_server(self):
        if self._ssh_client is not None:
            self._ssh_client.close()

        self._ssh_client = None
        self._remote_server_channel = None

    def kill_local_server(self, delay:int=0):
        """
        Kill the server process.
        :param delay: Seconds to wait for server to terminate itself.
        """
        assert delay >= 0
        for d in range(delay):
            if self._local_server_process is None or self._local_server_process.poll() is not None:
                return

            time.sleep(1)

        self._local_server_process.kill()

    def init(self, tiktorch_config, binary_model_file, binary_model_state, binary_optimizer_state):
        logger = logging.getLogger('TikTorchClient.init')
        # Make server for zmq
        logger.info("Setting up ZMQ Context...")
        self._zmq_context = zmq.Context()
        logger.info("Setting up ZMQ Socket...")
        self._zmq_socket = self._zmq_context.socket(zmq.PAIR)
        # set high water mark
        self._zmq_socket.set_hwm(16)
        info = logger.info("Connect to socket...")
        self._zmq_socket.connect(f'tcp://{self.addr}:{self.port}')
        # Send build directory

    def load_model(self, tiktorch_config, binary_model_file, binary_model_state, binary_optimizer_state):
        logger = logging.getLogger('TikTorchClient.init')
        logger.info("Sending init data...")
        self._zmq_socket.send_multipart([
            jsonapi.dumps({'id': 'INIT'}),
            jsonapi.dumps(tiktorch_config),
            binary_model_file,
            binary_model_state,
            binary_optimizer_state,
        ])
            # self._zmq_socket.send_json(tiktorch_config)
            # self._zmq_socket.send(binary_model_file)
            # self._zmq_socket.send(binary_model_state)
            # self._zmq_socket.send(binary_optimizer_state)

        logger.info("Init data sent.")
        return self

    def terminate(self):
        self._process.terminate()
        return self

    def kill(self):
        self._process.kill()
        return self

    def is_running(self):
        return self._process.poll() is None

    def get(self, tag, default=None, assert_exist=False):
        if assert_exist:
            assert tag in self._config, f"Tag '{tag}' not found in configuration."
        return self._config.get(tag, default)

    def meta_send(self, info_dict):
        self._zmq_socket.send_json(info_dict)
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
        assert framework in ('numpy', 'torch')
        tensor_spec = self.meta_recv()
        assert tensor_spec['id'] == f"{key.upper()}.TENSORSPEC"
        # Receive the buffer
        buf = memoryview(self._zmq_socket.recv())
        x = np.frombuffer(buf, dtype=tensor_spec['dtype'].lstrip('torch.')).reshape(tensor_spec['shape'])
        if framework == 'torch':
            x = torch.from_numpy(x).to(tensor_spec['device'])

        return x

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
            self.log_server_report()
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
            for idx, batch in enumerate(batches):
                logger.info("Sending batch.")
                self.tensor_send(batch, f'FORWARD_IN_{idx}')
            # Receive meta data
            output_tensor = self.tensor_recv('FORWARD_OUT')
            logger.info(f"Output received (shape = {tuple(output_tensor.shape)}).")
        return output_tensor

    def train(self, data, labels, sample_ids: Sequence[tuple]):
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
                    'labels.shapes': [tuple(_label.shape) for _label in labels],
                    'sample_ids': sample_ids}
            logger.info("Sending BatchSpec")
            self.meta_send(info)
            # Send tensors
            logger.info("Sending data and labels...")
            for _data, _label, id in zip(data, labels, sample_ids):
                self.tensor_send(_data, f'TRAIN_DATA_{id}')
                self.tensor_send(_label, f'TRAIN_LABEL_{id}')
            logger.info("Data and labels sent.")

    def set_hparams(self, hparams: dict):
        logger = logging.getLogger('TikTorchClient.set_hparams')
        logger.info("Waiting for Lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('HYPERPARAMETERS')
            logger.info("Request successful.")
            # Build info dict
            info = {'id': 'TRAIN.HYPERPARAMETERS',
                    'parameters': hparams}
            logger.info("Sending hyperparameters...")
            self.meta_send(info)

    def shutdown(self):
        logger = logging.getLogger('TikTorchClient.shutdown')
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('SHUTDOWN')
            logger.info("Request successful.")
            self.kill_server(delay=10)
            self._zmq_socket.close()

    def pause(self):
        logger = logging.getLogger('TikTorchClient.pause')
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('PAUSE')
            logger.info("Request successful.")
            self.log_server_report()

    def resume(self):
        logger = logging.getLogger('TikTorchClient.resume')
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('RESUME')
            logger.info("Request successful.")
            self.log_server_report()

    def training_process_is_running(self):
        logger = logging.getLogger("TikTorchClient.training_process_is_running")
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('POLL_TRAIN')
            # Receive info
            info = self.meta_recv()
        self.log_server_report()
        logger.info(f"return: {info['is_alive']}")
        return info['is_alive']

    def get_binary_model_state(self):
        logger = logging.getLogger('TikTorchClient.get_binary_model_state')
        logger.info('Requesting model state dict...')
        state_dict = self.request_model_state_dict()
        return pickle.dumps(state_dict)

    def request_model_state_dict(self):
        logger = logging.getLogger('TikTorchClient.request_model_state_dict')
        logger.info("Waiting for lock...")
        with self._main_lock:
            logger.info("Requesting dispatch...")
            assert self.request_dispatch('MODEL_STATE_DICT_REQUEST')
            logger.info("Request successful. Waiting for model state dict...")
            state_dict = self._zmq_socket.recv()
            logger.info("Model state dict received.")
        return state_dict

    def get_binary_optimizer_state(self):
        logger = logging.getLogger('TikTorchClient.get_binary_optimizer_state')
        logger.info('Requesting optimizer state dict...')
        state_dict = self.request_optimizer_state_dict()
        if state_dict is None:
            return b''
        else:
            return pickle.dumps(state_dict)

    def request_optimizer_state_dict(self):
        pass


def test_client_forward():
    TikTorchClient._START_PROCESS = False
    client = TikTorchClient(*INIT_DATA)
    logging.info("Obtained client. Forwarding...")
    out = client.forward([np.random.uniform(size=(256, 256)).astype('float32') for _ in range(1)])
    logging.info(f"out.shape = {out.shape}")
    logging.info("Forwarding again...")
    out = client.forward([np.random.uniform(size=(256, 256)).astype('float32') for _ in range(1)])
    logging.info(f"out.shape = {out.shape}")
    logging.info("Shutting down...")
    client.shutdown()
    logging.info("Done!")


def test_client_hparams():
    # Test for changing hyperparameter setting before and during training
    TikTorchClient._START_PROCESS = False
    client = TikTorchClient(*INIT_DATA)
    logging.info("Polling")
    is_running = client.training_process_is_running()
    logging.info(f"Training process running? {is_running}")

    client.set_hparams(dict(optimizer_kwargs=dict(lr=0.0003, weight_decay=0.0001, amsgrad=True),
                            optimizer_name='Adam',
                            criterion_kwargs=dict(reduce=False),
                            criterion_name='BCEWithLogitsLoss',
                            batch_size=1,
                            cache_size=200,
                            augmentor_kwargs={'invert_binary_labels': True}))

    logging.info("Sending train data and labels.")
    train_data = [np.random.uniform(size=(1, 64, 64)).astype('float32') for _ in range(10)]
    train_labels = [np.random.randint(0, 2, size=(1, 64, 64)).astype('float32') for _ in range(10)]
    client.train(train_data, train_labels, list(range(len(train_data))))
    logging.info("Sent train data and labels.")

    client.set_hparams(dict(optimizer_kwargs=dict(lr=0.0005, weight_decay=0.0002, amsgrad=True),
                            optimizer_name='Adam',
                            criterion_kwargs=dict(reduce=False),
                            criterion_name='BCEWithLogitsLoss',
                            batch_size=2,
                            cache_size=200,
                            augmentor_kwargs={'invert_binary_labels': True}))

    logging.info("Sending train data and labels...")
    train_data = [np.random.uniform(size=(1, 64, 64)).astype('float32') for _ in range(10)]
    train_labels = [np.random.randint(0, 2, size=(1, 64, 64)).astype('float32') for _ in range(10)]
    client.train(train_data, train_labels, list(range(len(train_data))))
    logging.info("Sent train data and labels.")

    client.set_hparams(dict(optimizer_kwargs=dict(lr=0.0005, weight_decay=0.0002, amsgrad=True),
                            optimizer_name='Adam',
                            criterion_kwargs=dict(reduce=False),
                            criterion_name='BCEWithLogitsLoss',
                            batch_size=3,
                            cache_size=200,
                            augmentor_kwargs={'invert_binary_labels': True}))

    logging.info("Waiting 10s...")
    time.sleep(10)

    logging.info("Shutting down...")
    client.shutdown()
    logging.info("Done!")


def test_client_train():
    TikTorchClient._START_PROCESS = False
    client = TikTorchClient(*INIT_DATA)
    logging.info("Obtained client. Forwarding...")
    out = client.forward([np.random.uniform(size=(256, 256)).astype('float32') for _ in range(1)])
    logging.info(f"out.shape = {out.shape}")

    logging.info("Polling")
    is_running = client.training_process_is_running()
    logging.info(f"Training process running? {is_running}")

    logging.info("Sending train data and labels...")
    train_data = [np.random.uniform(size=(1, 256, 256)).astype('float32') for _ in range(4)]
    train_labels = [np.random.randint(0, 2, size=(1, 256, 256)).astype('float32') for _ in range(4)]
    client.train(train_data, train_labels, list(range(len(train_data))))
    logging.info("Sent train data and labels and waiting for 15s...")

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


def test_client_state_request():
    import io
    import torch
    client = TikTorchClient(tiktorch_config=INIT_DATA[0],
                            binary_model_file=INIT_DATA[1],
                            binary_model_state=INIT_DATA[2])

    state_dict = client.request_model_state_dict()
    file = io.BytesIO(state_dict)
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)

    logging.info("Polling")
    is_running = client.training_process_is_running()
    logging.info(f"Training process running? {is_running}")

    logging.info("Sending train data and labels...")
    train_data = [np.random.uniform(size=(1, 128, 128)).astype('float32') for _ in range(10)]
    train_labels = [np.random.randint(0, 2, size=(1, 128, 128)).astype('float32') for _ in range(10)]
    client.train(train_data, train_labels, list(range(len(train_data))))
    logging.info("Sent train data and labels and waiting for 10s...")
    time.sleep(10)

    state_dict = client.request_model_state_dict()
    file = io.BytesIO(state_dict)
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)

    client.shutdown()


if __name__ == '__main__':
    import argparse
    import platform

    print('Python %s on %s' % (sys.version, sys.platform))

    parsey = argparse.ArgumentParser()
    parsey.add_argument('--name', type=str, default='jo')
    args = parsey.parse_args()

    BUILD_DIR = None
    if args.name == 'fynn':
        if platform.system() == 'Windows':
            BUILD_DIR = '/Users/fbeut/documents/ilastik/models/CREMI_DUNet_pretrained_new'
        else:
            BUILD_DIR = '/mnt/c/Users/fbeut/documents/ilastik/models/CREMI_DUNet_pretrained_new'
    elif args.name == 'jo':
        BUILD_DIR = '/home/jo/CREMI_DUNet_pretrained_new'

    INIT_DATA = []
    with open(os.path.join(BUILD_DIR, 'tiktorch_config.yml')) as file:
        INIT_DATA.append(yaml.load(file))

    for fn in ['model.py', 'state.nn']:
        with open(os.path.join(BUILD_DIR, fn), 'rb') as file:
            INIT_DATA.append(file.read())

    test_client_state_request()
