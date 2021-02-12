import os
import socket

import pytest

from tiktorch.launcher import ConnConf, LocalServerLauncher, RemoteSSHServerLauncher, SSHCred, client_factory

SSH_HOST_VAR = "TEST_SSH_HOST"
SSH_PORT_VAR = "TEST_SSH_PORT"
SSH_USER_VAR = "TEST_SSH_USER"
SSH_PWD_VAR = "TEST_SSH_PWD"
SSH_KEY_VAR = "TEST_SSH_KEY"


def test_start_local_server(srv_port):
    conn_conf = ConnConf("127.0.0.1", srv_port, timeout=5)
    launcher = LocalServerLauncher(conn_conf)
    launcher.start()

    assert launcher.is_server_running()

    client = client_factory(conn_conf)

    assert client.ping()

    launcher.stop()


def test_start_remote_server(srv_port):
    host, ssh_port = os.getenv(SSH_HOST_VAR), os.getenv(SSH_PORT_VAR, 22)
    user, pwd = os.getenv(SSH_USER_VAR), os.getenv(SSH_PWD_VAR)
    key = os.getenv(SSH_KEY_VAR)

    if not all([host, ssh_port, user, key or pwd]):
        pytest.skip(
            "To run this test specify "
            f"{SSH_HOST_VAR}, {SSH_USER_VAR} {SSH_PWD_VAR} or {SSH_KEY_VAR} and optionaly {SSH_PORT_VAR}"
        )

    conn_conf = ConnConf(socket.gethostbyname(host), srv_port, timeout=20)
    cred = SSHCred(user=user, password=pwd, key_path=key)
    launcher = RemoteSSHServerLauncher(conn_conf, cred=cred)

    client = client_factory(conn_conf)
    try:
        launcher.start()

        assert launcher.is_server_running()

        assert client.ping()
    finally:
        launcher.stop()

    assert not launcher.is_server_running()
