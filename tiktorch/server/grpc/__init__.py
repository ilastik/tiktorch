import threading
from concurrent import futures

import grpc

from tiktorch.server.device_pool import TorchDevicePool
from tiktorch.server.session_manager import SessionManager
from tiktorch.server.data_store import DataStore

from .inference_servicer import InferenceServicer
from .flight_control_servicer import FlightControlServicer
from .data_store_servicer import DataStoreServicer

import inference_pb2_grpc
import data_store_pb2_grpc


def serve(host, port):
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

    inference_svc = InferenceServicer(TorchDevicePool(), SessionManager(), data_store)
    fligh_svc = FlightControlServicer(done_evt=done_evt)
    data_svc = DataStoreServicer(data_store)

    inference_pb2_grpc.add_InferenceServicer_to_server(inference_svc, server)
    inference_pb2_grpc.add_FlightControlServicer_to_server(fligh_svc, server)
    data_store_pb2_grpc.add_DataStoreServicer_to_server(data_svc, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    done_evt.wait()

    server.stop(0).wait()
