import json
import os
import threading
from concurrent import futures
from typing import Optional

import grpc

from tiktorch.proto import data_store_pb2_grpc, inference_pb2_grpc, training_pb2_grpc
from tiktorch.server.data_store import DataStore
from tiktorch.server.device_pool import IDevicePool, TorchDevicePool
from tiktorch.server.session_manager import SessionManager

from .data_store_servicer import DataStoreServicer
from .flight_control_servicer import FlightControlServicer
from .inference_servicer import InferenceServicer
from .training_servicer import TrainingServicer


def _print_available_devices(device_pool: IDevicePool) -> None:
    cuda = device_pool.cuda_version
    print()
    print("CUDA version:", cuda or "not available")
    for env_var, value in os.environ.items():
        if env_var.startswith("CUDA_"):
            print(env_var, value.strip() or "<empty>")

    print()
    print("Available devices:")
    for device in device_pool.list_devices():
        print(f"  * {device.id}")


def serve(host, port, *, connection_file_path: Optional[str] = None, kill_timeout: Optional[float] = None):
    """
    Starts grpc server on given host and port and writes connection details to json file
    :param host: ip to listen on
    :param port: port to listen on (if 0 random port will be assigned)
    :param connection_file_path: path to file where to write connection parameters
    :param kill_timeout: how long to wait for heartbeat before stopping server
    """
    _100_MB = 100 * 1024 * 1024

    done_evt = threading.Event()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=32),
        options=[
            ("grpc.max_send_message_length", _100_MB),
            ("grpc.max_receive_message_length", _100_MB),
            ("grpc.so_reuseport", 0),
        ],
    )

    data_store = DataStore()
    device_pool = TorchDevicePool()

    inference_svc = InferenceServicer(device_pool, SessionManager(), data_store)
    fligh_svc = FlightControlServicer(done_evt=done_evt, kill_timeout=kill_timeout)
    data_svc = DataStoreServicer(data_store)

    training_svc = TrainingServicer(device_pool=device_pool, session_manager=SessionManager())

    _print_available_devices(device_pool)

    inference_pb2_grpc.add_InferenceServicer_to_server(inference_svc, server)
    inference_pb2_grpc.add_FlightControlServicer_to_server(fligh_svc, server)
    data_store_pb2_grpc.add_DataStoreServicer_to_server(data_svc, server)
    training_pb2_grpc.add_TrainingServicer_to_server(training_svc, server)

    acquired_port = server.add_insecure_port(f"{host}:{port}")
    print()
    print(f"Starting server on {host}:{acquired_port}")
    if connection_file_path:
        print(f"Writing connection data to {connection_file_path}")
        with open(connection_file_path, "w") as conn_file:
            json.dump({"addr": host, "port": acquired_port}, conn_file)

    server.start()

    done_evt.wait()

    server.stop(0).wait()
