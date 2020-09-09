import grpc

import inference_pb2
import inference_pb2_grpc


class FlightControlServicer(inference_pb2_grpc.FlightControlServicer):
    def __init__(self, *, done_evt=None) -> None:
        self.__done_evt = done_evt

    def Ping(self, request: inference_pb2.Empty, context) -> inference_pb2.Empty:
        return inference_pb2.Empty()

    def Shutdown(self, request: inference_pb2.Empty, context) -> inference_pb2.Empty:
        if self.__done_evt:
            self.__done_evt.set()
        return inference_pb2.Empty()
