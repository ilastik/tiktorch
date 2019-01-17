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
from paramiko import SSHClient, AutoAddPolicy
from socket import gethostbyname, timeout

import tiktorch.utils as utils
from tiktorch.tio import TikIn

logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    torch.multiprocessing.set_start_method('spawn', force=True)


class TikTorchClient(object):
    RANK = 0
    SIZE = 2

    _START_PROCESS = False

    def __init__(self, local_build_dir=None, remote_build_dir=None, remote_model_dir=None, address='localhost',
                 ssh_port=22, port='5556', meta_port='5557', ilp_directory=None, start_server=True, username=None,
                 password=None):
        assert local_build_dir is not None or remote_build_dir is not None
        assert local_build_dir is None or remote_build_dir is None
        assert remote_build_dir is None or remote_model_dir is None
        # todo: actually use meta_port
        self.local_build_dir = local_build_dir
        self.remote_build_dir = remote_build_dir
        self.remote_model_dir = remote_model_dir
        logger = logging.getLogger()
        logger.info(f'local build directory {local_build_dir}')
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
        self.ilp_directory = ilp_directory  # todo: remove ilp_directory
        # Privates
        self._args: list = None
        self._process: subprocess.Popen = None
        self._config = {}
        self._zmq_context = None
        self._zmq_socket = None
        # Locks
        self._main_lock = thr.Lock()
        self._zmq_lock = thr.Lock()
        # Initialize
        self.read_config()
        self._local_server_process = None
        self._ssh_client = None
        self._remote_server_channel = None
        if start_server:
            self.start_server()

        self.init()

    def start_server(self):
        with self._main_lock:
            if self.addr in ('127.0.0.1', 'localhost'):
                self.remote_build_dir = self.local_build_dir
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

                if self.remote_build_dir is None:
                    assert self.local_build_dir is not None
                    # transfer local_build_dir to server
                    # todo: do with zmq instead?
                    build_dir_name = os.path.basename(self.local_build_dir)
                    logger.info('Opening sftp connection...')
                    sftp = self._ssh_client.open_sftp()
                    if self.remote_model_dir is None:
                        remote_cwd = f'/home/{self.ssh_connect["username"]}'
                    else:
                        remote_cwd = self.remote_model_directory

                    logger.info(f'Setting remote cwd to {remote_cwd} ...')
                    sftp.chdir(remote_cwd)
                    logger.info(f'Creating remote build directory "{build_dir_name}"...')
                    try:
                        sftp.mkdir(build_dir_name)
                    except Exception:
                        logger.debug(f'Failed to create remote build directory. Does it already exist?')

                    try:
                        for root, dirs, files in os.walk(self.local_build_dir):
                            for file in files:
                                if file.endswith('.pyc'):
                                    continue

                                def cb(done:int, total:int):
                                    logger.info(f'Transfering {os.path.join(build_dir_name, file)} {done/total*100:2.0f}%')

                                sftp.put(os.path.join(root, file), os.path.join(build_dir_name, file), callback=cb)

                        self.remote_build_dir = build_dir_name
                    except timeout as e:
                        logger.error(
                            f'Could not transfer content of local building directory {self.local_build_dir}\n{e}')
                        self.kill_remote_server()
                        return

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

    def init(self):
        logger = logging.getLogger('TikTorchClient.init')
        # Make server for zmq
        logger.info("Setting up ZMQ Context...")
        self._zmq_context = zmq.Context()
        logger.info("Setting up ZMQ Socket...")
        self._zmq_socket = self._zmq_context.socket(zmq.PAIR)
        logger.info("Connect to socket...")
        self._zmq_socket.connect(f'tcp://{self.addr}:{self.port}')
        # Send build directory
        logger.info("Sending build directory...")
        self.meta_send({'id': 'INIT.PATHS',
                        'build_dir': self.remote_build_dir,
                        'ilp_dir': self.ilp_directory})
        logger.info("Build directory sent.")
        self.log_server_report()
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
        config_file_name = os.path.join(self.local_build_dir, 'tiktorch_config.yml')
        if not os.path.exists(config_file_name):
            raise FileNotFoundError(f"Config file not found in "
                                    f"local_build_dir: {self.local_build_dir}.")
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

    def train(self, data, labels, sample_ids: list):
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
    client = TikTorchClient(local_build_dir='.', address='127.0.0.1', port='29500',
                            meta_port='29501')
    client._model = torch.nn.Conv2d(1, 1, 1)
    client._config = {'input_shape': [1, 512, 512],
                      'dynamic_input_shape': '(32 * (nH + 1), 32 * (nW + 1))'}
    return client


# hacky lazy global build dir for tests
ILP_DIR = None
BUILD_DIR = None
import platform

if platform.system() == 'Windows':
    BUILD_DIR = '/Users/fbeut/documents/ilastik/models/CREMI_DUNet_pretrained_new'
else:
    BUILD_DIR = '/mnt/c/Users/fbeut/documents/ilastik/models/CREMI_DUNet_pretrained_new'


def test_client_forward():
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


def test_client_hparams():
    # Test for changing hyperparameter setting before and during training
    TikTorchClient._START_PROCESS = False
    client = TikTorchClient(BUILD_DIR)
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
    client = TikTorchClient(BUILD_DIR, ilp_directory=ILP_DIR)
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


if __name__ == '__main__':
    print('Python %s on %s' % (sys.version, sys.platform))
    test_client_forward()
    test_client_train()
    test_client_hparams()
