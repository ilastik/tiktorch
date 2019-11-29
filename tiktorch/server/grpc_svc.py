import time
from concurrent import futures

import grpc

from tiktorch.proto import inference_pb2, inference_pb2_grpc

_ONE_DAY_IN_SECONDS = 24 * 60 * 60


class SessionServicer(inference_pb2_grpc.SessionProviderServicer):
    def GetDevices(self, request: inference_pb2.Empty, context) -> inference_pb2.Devices:
        return inference_pb2.Devices(names=["cpu", "gpu:0"])


def serve(host, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_SessionProviderServicer_to_server(SessionServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
