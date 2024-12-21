# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import training_pb2 as training__pb2
from . import utils_pb2 as utils__pb2


class TrainingStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ListDevices = channel.unary_unary(
                '/training.Training/ListDevices',
                request_serializer=utils__pb2.Empty.SerializeToString,
                response_deserializer=utils__pb2.Devices.FromString,
                )
        self.Init = channel.unary_unary(
                '/training.Training/Init',
                request_serializer=training__pb2.TrainingConfig.SerializeToString,
                response_deserializer=utils__pb2.ModelSession.FromString,
                )
        self.Start = channel.unary_unary(
                '/training.Training/Start',
                request_serializer=utils__pb2.ModelSession.SerializeToString,
                response_deserializer=utils__pb2.Empty.FromString,
                )
        self.Resume = channel.unary_unary(
                '/training.Training/Resume',
                request_serializer=utils__pb2.ModelSession.SerializeToString,
                response_deserializer=utils__pb2.Empty.FromString,
                )
        self.Pause = channel.unary_unary(
                '/training.Training/Pause',
                request_serializer=utils__pb2.ModelSession.SerializeToString,
                response_deserializer=utils__pb2.Empty.FromString,
                )
        self.StreamUpdates = channel.unary_stream(
                '/training.Training/StreamUpdates',
                request_serializer=utils__pb2.ModelSession.SerializeToString,
                response_deserializer=training__pb2.StreamUpdateResponse.FromString,
                )
        self.GetLogs = channel.unary_unary(
                '/training.Training/GetLogs',
                request_serializer=utils__pb2.ModelSession.SerializeToString,
                response_deserializer=training__pb2.GetLogsResponse.FromString,
                )
        self.Export = channel.unary_unary(
                '/training.Training/Export',
                request_serializer=utils__pb2.ModelSession.SerializeToString,
                response_deserializer=utils__pb2.Empty.FromString,
                )
        self.Predict = channel.unary_unary(
                '/training.Training/Predict',
                request_serializer=utils__pb2.PredictRequest.SerializeToString,
                response_deserializer=utils__pb2.PredictResponse.FromString,
                )
        self.GetStatus = channel.unary_unary(
                '/training.Training/GetStatus',
                request_serializer=utils__pb2.ModelSession.SerializeToString,
                response_deserializer=training__pb2.GetStatusResponse.FromString,
                )
        self.CloseTrainerSession = channel.unary_unary(
                '/training.Training/CloseTrainerSession',
                request_serializer=utils__pb2.ModelSession.SerializeToString,
                response_deserializer=utils__pb2.Empty.FromString,
                )


class TrainingServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ListDevices(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Init(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Start(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Resume(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Pause(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamUpdates(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetLogs(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Export(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetStatus(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CloseTrainerSession(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TrainingServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ListDevices': grpc.unary_unary_rpc_method_handler(
                    servicer.ListDevices,
                    request_deserializer=utils__pb2.Empty.FromString,
                    response_serializer=utils__pb2.Devices.SerializeToString,
            ),
            'Init': grpc.unary_unary_rpc_method_handler(
                    servicer.Init,
                    request_deserializer=training__pb2.TrainingConfig.FromString,
                    response_serializer=utils__pb2.ModelSession.SerializeToString,
            ),
            'Start': grpc.unary_unary_rpc_method_handler(
                    servicer.Start,
                    request_deserializer=utils__pb2.ModelSession.FromString,
                    response_serializer=utils__pb2.Empty.SerializeToString,
            ),
            'Resume': grpc.unary_unary_rpc_method_handler(
                    servicer.Resume,
                    request_deserializer=utils__pb2.ModelSession.FromString,
                    response_serializer=utils__pb2.Empty.SerializeToString,
            ),
            'Pause': grpc.unary_unary_rpc_method_handler(
                    servicer.Pause,
                    request_deserializer=utils__pb2.ModelSession.FromString,
                    response_serializer=utils__pb2.Empty.SerializeToString,
            ),
            'StreamUpdates': grpc.unary_stream_rpc_method_handler(
                    servicer.StreamUpdates,
                    request_deserializer=utils__pb2.ModelSession.FromString,
                    response_serializer=training__pb2.StreamUpdateResponse.SerializeToString,
            ),
            'GetLogs': grpc.unary_unary_rpc_method_handler(
                    servicer.GetLogs,
                    request_deserializer=utils__pb2.ModelSession.FromString,
                    response_serializer=training__pb2.GetLogsResponse.SerializeToString,
            ),
            'Export': grpc.unary_unary_rpc_method_handler(
                    servicer.Export,
                    request_deserializer=utils__pb2.ModelSession.FromString,
                    response_serializer=utils__pb2.Empty.SerializeToString,
            ),
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=utils__pb2.PredictRequest.FromString,
                    response_serializer=utils__pb2.PredictResponse.SerializeToString,
            ),
            'GetStatus': grpc.unary_unary_rpc_method_handler(
                    servicer.GetStatus,
                    request_deserializer=utils__pb2.ModelSession.FromString,
                    response_serializer=training__pb2.GetStatusResponse.SerializeToString,
            ),
            'CloseTrainerSession': grpc.unary_unary_rpc_method_handler(
                    servicer.CloseTrainerSession,
                    request_deserializer=utils__pb2.ModelSession.FromString,
                    response_serializer=utils__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'training.Training', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Training(object):
    """Missing associated documentation comment in .proto file."""

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
        return grpc.experimental.unary_unary(request, target, '/training.Training/ListDevices',
            utils__pb2.Empty.SerializeToString,
            utils__pb2.Devices.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Init(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/training.Training/Init',
            training__pb2.TrainingConfig.SerializeToString,
            utils__pb2.ModelSession.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Start(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/training.Training/Start',
            utils__pb2.ModelSession.SerializeToString,
            utils__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Resume(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/training.Training/Resume',
            utils__pb2.ModelSession.SerializeToString,
            utils__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Pause(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/training.Training/Pause',
            utils__pb2.ModelSession.SerializeToString,
            utils__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StreamUpdates(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/training.Training/StreamUpdates',
            utils__pb2.ModelSession.SerializeToString,
            training__pb2.StreamUpdateResponse.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/training.Training/GetLogs',
            utils__pb2.ModelSession.SerializeToString,
            training__pb2.GetLogsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Export(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/training.Training/Export',
            utils__pb2.ModelSession.SerializeToString,
            utils__pb2.Empty.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/training.Training/Predict',
            utils__pb2.PredictRequest.SerializeToString,
            utils__pb2.PredictResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetStatus(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/training.Training/GetStatus',
            utils__pb2.ModelSession.SerializeToString,
            training__pb2.GetStatusResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CloseTrainerSession(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/training.Training/CloseTrainerSession',
            utils__pb2.ModelSession.SerializeToString,
            utils__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
