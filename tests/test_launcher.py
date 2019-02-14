import os

import pytest

from tiktorch.launcher import LocalServerLauncher, RemoteSSHServerLauncher, wait
from tiktorch.rpc import Client, TimeoutError
from tiktorch.rpc_interface import IFlightControl

SSH_HOST_VAR = "TEST_SSH_HOST"
SSH_PORT_VAR = "TEST_SSH_PORT"
SSH_USER_VAR = "TEST_SSH_USER"
SSH_PWD_VAR = "TEST_SSH_PWD"


def test_start_local_server(srv_port):
    launcher = LocalServerLauncher()
    addr = '127.0.0.1'
    launcher.start(addr, srv_port)

    assert launcher.is_server_running()

    client = Client(IFlightControl(), f'tcp://{addr}:{srv_port}')

    assert client.ping() == b'pong'

    client.shutdown()

    launcher.is_server_running()


def test_start_remote_server(srv_port):
    host, port = os.getenv(SSH_HOST_VAR), os.getenv(SSH_PORT_VAR, 22)
    user, pwd = os.getenv(SSH_USER_VAR), os.getenv(SSH_PWD_VAR)

    if not all([host, port, user, pwd]):
        print([host, port, user, pwd])
        pytest.skip(
            "To run this test specify "
            f"{SSH_HOST_VAR}, {SSH_USER_VAR} {SSH_PWD_VAR} and optionaly {SSH_PORT_VAR}"
        )

    launcher = RemoteSSHServerLauncher(user=user, password=pwd)
    launcher.start(host, srv_port)

    wait(launcher.is_server_running, max_wait=2)

    client = Client(IFlightControl(), f'tcp://{host}:{srv_port}')

    assert client.ping() == b'pong'
    client.shutdown()

    wait(lambda: not launcher.is_server_running(), max_wait=1)
