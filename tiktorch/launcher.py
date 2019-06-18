import sys
import time
import enum
import logging
import subprocess
import threading

from socket import timeout
from typing import Optional

from paramiko import SSHClient, AutoAddPolicy

from .rpc import Client, Timeout, TCPConnConf, Shutdown
from .rpc_interface import IFlightControl


HEARTBEAT_INTERVAL = 10  # seconds
KILL_TIMEOUT = 60  # seconds


class AlreadyRunningError(Exception):
    pass


class IServerLauncher:
    class State(enum.Enum):
        Stopped = "Stopped"
        Running = "Running"

    __conn_conf = None
    _state: State = State.Stopped
    _heartbeat_worker: Optional[threading.Thread] = None

    @property
    def logger(self):
        return logging.getLogger(self.__class__.__qualname__)

    @property
    def _conn_conf(self) -> TCPConnConf:
        if self.__conn_conf is None:
            raise Exception("Please set self._conn_conf")

        return self.__conn_conf

    @_conn_conf.setter
    def _conn_conf(self, value: TCPConnConf) -> None:
        if not isinstance(value, TCPConnConf):
            raise ValueError("Should be instance of TCPConnConf")

        self.__conn_conf = value

    def _hearbeat(self, interval: int):
        while not self._stop.wait(timeout=interval):
            if not self._ping():
                self.logger.debug("Server has failed to respond to a ping")

    def _start_server(self, dummy: bool, kill_timeout: int):
        raise NotImplementedError

    def _ping(self):
        try:
            c = Client(IFlightControl(), self._conn_conf)
            return c.ping() == b"pong"
        except Timeout:
            return False

    def is_server_running(self):
        return self._ping()

    def start(self, ping_interval: int = HEARTBEAT_INTERVAL, kill_timeout: int = KILL_TIMEOUT, *, dummy: bool = False):
        self._start_server(dummy, kill_timeout)
        self._state = self.State.Running

        self._stop = threading.Event()
        self._heartbeat_worker = threading.Thread(
            target=self._hearbeat, args=(ping_interval,), name=f"{type(self).__name__}.HeartbeatWorker"
        )
        self._heartbeat_worker.daemon = True
        self._heartbeat_worker.start()

    def stop(self):
        if self._state == self.State.Stopped:
            raise Exception("Server has already been stopped")

        self._stop.set()

        c = Client(IFlightControl(), self._conn_conf)
        try:
            c.shutdown()
        except Shutdown:
            pass

        self._state = self.State.Stopped
        self._heartbeat_worker.join()


def wait(done, interval=0.1, max_wait=10):
    start = time.time()

    while True:
        if done():
            break

        passed = time.time() - start

        if passed > max_wait:
            raise Timeout()
        else:
            time.sleep(interval)


class LocalServerLauncher(IServerLauncher):
    def __init__(self, conn_conf: TCPConnConf):
        self._conn_conf = conn_conf
        self._process = None

    def _start_server(self, dummy: bool, kill_timeout: int):
        addr, port, notify_port = self._conn_conf.addr, self._conn_conf.port, self._conn_conf.pubsub_port

        if addr != "127.0.0.1":
            raise ValueError("LocalServerHandler only possible to run on localhost")

        if self._process:
            raise AlreadyRunningError(f"Local server is already running (pid:{self._process.pid})")

        self.logger.info("Starting local TikTorchServer on %s:%s", addr, port)

        cmd = [
            sys.executable,
            "-m",
            "tiktorch.server",
            "--port",
            str(port),
            "--notify-port",
            str(notify_port),
            "--addr",
            addr,
            "--kill-timeout",
            str(kill_timeout),
        ]
        if dummy:
            cmd.append("--dummy")

        self._process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

        try:
            wait(self.is_server_running)
        except Timeout:
            raise Exception("Failed to start local TikTorchServer")


class SSHCred:
    def __init__(self, user: str, password: Optional[str] = None, key_path: Optional[str] = None) -> None:
        self.user = user
        self.password = password
        self.key_path = key_path

    def __repr__(self) -> str:
        return f"<SSHCred({self.user}, has_pwd={bool(self.password)}, key={self.key_path})>"


class RemoteSSHServerLauncher(IServerLauncher):
    def __init__(self, conn_conf: TCPConnConf, *, cred: SSHCred, ssh_port: int = 22) -> None:

        self._ssh_port = ssh_port
        self._channel = None
        self._conn_conf = conn_conf
        self._cred = cred

        self._setup_ssh_client()

    def _setup_ssh_client(self):
        self._ssh_client = SSHClient()
        self._ssh_client.set_missing_host_key_policy(AutoAddPolicy())
        self._ssh_client.load_system_host_keys()

    def _start_server(self, dummy: bool, kill_timeout: int):
        if self._channel:
            raise RuntimeError("SSH server is already running")

        addr, port, notify_port = self._conn_conf.addr, self._conn_conf.port, self._conn_conf.pubsub_port

        ssh_params = {
            "hostname": addr,
            "port": self._ssh_port,
            "username": self._cred.user,
            "password": self._cred.password,
            "key_filename": self._cred.key_path,
            "timeout": 10,
        }

        try:
            self._ssh_client.connect(**ssh_params)
        except timeout as e:
            raise RuntimeError("Failed to establish SSH connection")

        transport = self._ssh_client.get_transport()

        channel = transport.open_session()
        channel.set_combine_stderr(True)
        buf_rdy = threading.Event()
        channel.in_buffer.set_event(buf_rdy)

        def _monitor_and_report():
            should_continue = True
            while should_continue:
                if channel.status_event.wait(timeout=1):
                    should_continue = False

                if buf_rdy.wait(timeout=1):
                    buf = b""
                    while channel.recv_ready():
                        buf += channel.recv(2048)

                    buf_rdy.clear()

                    if buf:
                        print(buf.decode("utf-8"))

                if not should_continue:
                    print("Server exited with status: %s" % channel.recv_exit_status())
                    transport.close()

        t = threading.Thread(target=_monitor_and_report, name="LauncherSSHMonitoring")
        t.start()

        self._channel = channel

        try:
            cmd = f"tiktorch --addr {addr} --port {port} --notify-port {notify_port} --kill-timeout {kill_timeout}"
            if dummy:
                cmd += " --dummy"

            channel.exec_command(cmd)
        except timeout as e:
            raise RuntimeError("Failed to start TiktorchServer")
