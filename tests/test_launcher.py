import os

import pytest
import socket

from tiktorch.launcher import LocalServerLauncher, RemoteSSHServerLauncher, wait
from tiktorch.rpc import Client, TCPConnConf
from tiktorch.rpc_interface import IFlightControl

SSH_HOST_VAR = "TEST_SSH_HOST"
SSH_PORT_VAR = "TEST_SSH_PORT"
SSH_USER_VAR = "TEST_SSH_USER"
SSH_PWD_VAR = "TEST_SSH_PWD"


def test_start_local_server(srv_port):
    conn_conf = TCPConnConf('127.0.0.1', srv_port, timeout=20)
    launcher = LocalServerLauncher(conn_conf)
    launcher.start()

    assert launcher.is_server_running()

    client = Client(IFlightControl(), conn_conf)

    assert client.ping() == b'pong'

    client.shutdown()

    launcher.is_server_running()


def test_start_remote_server(srv_port):
    print(os.environ)
    host, ssh_port = os.getenv(SSH_HOST_VAR), os.getenv(SSH_PORT_VAR, 22)
    conn_conf = TCPConnConf(socket.gethostbyname(host), srv_port, timeout=20)
    user, pwd = os.getenv(SSH_USER_VAR), os.getenv(SSH_PWD_VAR)

    if not all([host, ssh_port, user, pwd]):
        print([host, ssh_port, user, pwd])
        pytest.skip(
            "To run this test specify "
            f"{SSH_HOST_VAR}, {SSH_USER_VAR} {SSH_PWD_VAR} and optionaly {SSH_PORT_VAR}"
        )

    launcher = RemoteSSHServerLauncher(conn_conf, user=user, password=pwd)
    launcher.start()

    wait(launcher.is_server_running, max_wait=2)

    client = Client(IFlightControl(), conn_conf)

    assert client.ping() == b'pong'

    client.shutdown()

    wait(lambda: not launcher.is_server_running(), max_wait=1)
