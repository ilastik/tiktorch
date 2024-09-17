# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import inference_pb2 as inference__pb2


class InferenceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CreateModelSession = channel.unary_unary(
                '/Inference/CreateModelSession',
                request_serializer=inference__pb2.CreateModelSessionRequest.SerializeToString,
                response_deserializer=inference__pb2.ModelSession.FromString,
                )
        self.CloseModelSession = channel.unary_unary(
                '/Inference/CloseModelSession',
                request_serializer=inference__pb2.ModelSession.SerializeToString,
                response_deserializer=inference__pb2.Empty.FromString,
                )
        self.CreateDatasetDescription = channel.unary_unary(
                '/Inference/CreateDatasetDescription',
                request_serializer=inference__pb2.CreateDatasetDescriptionRequest.SerializeToString,
                response_deserializer=inference__pb2.DatasetDescription.FromString,
                )
        self.GetLogs = channel.unary_stream(
                '/Inference/GetLogs',
                request_serializer=inference__pb2.Empty.SerializeToString,
                response_deserializer=inference__pb2.LogEntry.FromString,
                )
        self.ListDevices = channel.unary_unary(
                '/Inference/ListDevices',
                request_serializer=inference__pb2.Empty.SerializeToString,
                response_deserializer=inference__pb2.Devices.FromString,
                )
        self.Predict = channel.unary_unary(
                '/Inference/Predict',
                request_serializer=inference__pb2.PredictRequest.SerializeToString,
                response_deserializer=inference__pb2.PredictResponse.FromString,
                )
        self.IsCudaOutOfMemory = channel.unary_unary(
                '/Inference/IsCudaOutOfMemory',
                request_serializer=inference__pb2.IsCudaOutOfMemoryRequest.SerializeToString,
                response_deserializer=inference__pb2.IsCudaOutOfMemoryResponse.FromString,
                )
        self.MaxCudaMemoryShape = channel.unary_unary(
                '/Inference/MaxCudaMemoryShape',
                request_serializer=inference__pb2.MaxCudaMemoryShapeRequest.SerializeToString,
                response_deserializer=inference__pb2.MaxCudaMemoryShapeResponse.FromString,
                )


class InferenceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CreateModelSession(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CloseModelSession(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateDatasetDescription(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetLogs(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListDevices(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def IsCudaOutOfMemory(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MaxCudaMemoryShape(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CreateModelSession': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateModelSession,
                    request_deserializer=inference__pb2.CreateModelSessionRequest.FromString,
                    response_serializer=inference__pb2.ModelSession.SerializeToString,
            ),
            'CloseModelSession': grpc.unary_unary_rpc_method_handler(
                    servicer.CloseModelSession,
                    request_deserializer=inference__pb2.ModelSession.FromString,
                    response_serializer=inference__pb2.Empty.SerializeToString,
            ),
            'CreateDatasetDescription': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateDatasetDescription,
                    request_deserializer=inference__pb2.CreateDatasetDescriptionRequest.FromString,
                    response_serializer=inference__pb2.DatasetDescription.SerializeToString,
            ),
            'GetLogs': grpc.unary_stream_rpc_method_handler(
                    servicer.GetLogs,
                    request_deserializer=inference__pb2.Empty.FromString,
                    response_serializer=inference__pb2.LogEntry.SerializeToString,
            ),
            'ListDevices': grpc.unary_unary_rpc_method_handler(
                    servicer.ListDevices,
                    request_deserializer=inference__pb2.Empty.FromString,
                    response_serializer=inference__pb2.Devices.SerializeToString,
            ),
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=inference__pb2.PredictRequest.FromString,
                    response_serializer=inference__pb2.PredictResponse.SerializeToString,
            ),
            'IsCudaOutOfMemory': grpc.unary_unary_rpc_method_handler(
                    servicer.IsCudaOutOfMemory,
                    request_deserializer=inference__pb2.IsCudaOutOfMemoryRequest.FromString,
                    response_serializer=inference__pb2.IsCudaOutOfMemoryResponse.SerializeToString,
            ),
            'MaxCudaMemoryShape': grpc.unary_unary_rpc_method_handler(
                    servicer.MaxCudaMemoryShape,
                    request_deserializer=inference__pb2.MaxCudaMemoryShapeRequest.FromString,
                    response_serializer=inference__pb2.MaxCudaMemoryShapeResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Inference', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Inference(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CreateModelSession(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Inference/CreateModelSession',
            inference__pb2.CreateModelSessionRequest.SerializeToString,
            inference__pb2.ModelSession.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CloseModelSession(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Inference/CloseModelSession',
            inference__pb2.ModelSession.SerializeToString,
            inference__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CreateDatasetDescription(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Inference/CreateDatasetDescription',
            inference__pb2.CreateDatasetDescriptionRequest.SerializeToString,
            inference__pb2.DatasetDescription.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetLogs(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/Inference/GetLogs',
            inference__pb2.Empty.SerializeToString,
            inference__pb2.LogEntry.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListDevices(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Inference/ListDevices',
            inference__pb2.Empty.SerializeToString,
            inference__pb2.Devices.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Inference/Predict',
            inference__pb2.PredictRequest.SerializeToString,
            inference__pb2.PredictResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def IsCudaOutOfMemory(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Inference/IsCudaOutOfMemory',
            inference__pb2.IsCudaOutOfMemoryRequest.SerializeToString,
            inference__pb2.IsCudaOutOfMemoryResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MaxCudaMemoryShape(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Inference/MaxCudaMemoryShape',
            inference__pb2.MaxCudaMemoryShapeRequest.SerializeToString,
            inference__pb2.MaxCudaMemoryShapeResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class FlightControlStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Ping = channel.unary_unary(
                '/FlightControl/Ping',
                request_serializer=inference__pb2.Empty.SerializeToString,
                response_deserializer=inference__pb2.Empty.FromString,
                )
        self.Shutdown = channel.unary_unary(
                '/FlightControl/Shutdown',
                request_serializer=inference__pb2.Empty.SerializeToString,
                response_deserializer=inference__pb2.Empty.FromString,
                )


class FlightControlServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Ping(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Shutdown(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FlightControlServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Ping': grpc.unary_unary_rpc_method_handler(
                    servicer.Ping,
                    request_deserializer=inference__pb2.Empty.FromString,
                    response_serializer=inference__pb2.Empty.SerializeToString,
            ),
            'Shutdown': grpc.unary_unary_rpc_method_handler(
                    servicer.Shutdown,
                    request_deserializer=inference__pb2.Empty.FromString,
                    response_serializer=inference__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'FlightControl', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FlightControl(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/FlightControl/Ping',
            inference__pb2.Empty.SerializeToString,
            inference__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Shutdown(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/FlightControl/Shutdown',
            inference__pb2.Empty.SerializeToString,
            inference__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
