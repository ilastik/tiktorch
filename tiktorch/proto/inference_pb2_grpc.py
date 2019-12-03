# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import inference_pb2 as inference__pb2


class InferenceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.ListDevices = channel.unary_unary(
        '/Inference/ListDevices',
        request_serializer=inference__pb2.Empty.SerializeToString,
        response_deserializer=inference__pb2.Devices.FromString,
        )
    self.CreateSession = channel.unary_unary(
        '/Inference/CreateSession',
        request_serializer=inference__pb2.Empty.SerializeToString,
        response_deserializer=inference__pb2.Session.FromString,
        )
    self.UseDevices = channel.unary_unary(
        '/Inference/UseDevices',
        request_serializer=inference__pb2.Devices.SerializeToString,
        response_deserializer=inference__pb2.Devices.FromString,
        )
    self.HasSession = channel.unary_unary(
        '/Inference/HasSession',
        request_serializer=inference__pb2.Session.SerializeToString,
        response_deserializer=inference__pb2.Session.FromString,
        )
    self.CloseSession = channel.unary_unary(
        '/Inference/CloseSession',
        request_serializer=inference__pb2.Session.SerializeToString,
        response_deserializer=inference__pb2.Empty.FromString,
        )
    self.Predict = channel.unary_unary(
        '/Inference/Predict',
        request_serializer=inference__pb2.PredictRequest.SerializeToString,
        response_deserializer=inference__pb2.PredictResponse.FromString,
        )
    self.GetLogs = channel.unary_stream(
        '/Inference/GetLogs',
        request_serializer=inference__pb2.Empty.SerializeToString,
        response_deserializer=inference__pb2.LogEntry.FromString,
        )


class InferenceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def ListDevices(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def CreateSession(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def UseDevices(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def HasSession(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def CloseSession(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Predict(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetLogs(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_InferenceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'ListDevices': grpc.unary_unary_rpc_method_handler(
          servicer.ListDevices,
          request_deserializer=inference__pb2.Empty.FromString,
          response_serializer=inference__pb2.Devices.SerializeToString,
      ),
      'CreateSession': grpc.unary_unary_rpc_method_handler(
          servicer.CreateSession,
          request_deserializer=inference__pb2.Empty.FromString,
          response_serializer=inference__pb2.Session.SerializeToString,
      ),
      'UseDevices': grpc.unary_unary_rpc_method_handler(
          servicer.UseDevices,
          request_deserializer=inference__pb2.Devices.FromString,
          response_serializer=inference__pb2.Devices.SerializeToString,
      ),
      'HasSession': grpc.unary_unary_rpc_method_handler(
          servicer.HasSession,
          request_deserializer=inference__pb2.Session.FromString,
          response_serializer=inference__pb2.Session.SerializeToString,
      ),
      'CloseSession': grpc.unary_unary_rpc_method_handler(
          servicer.CloseSession,
          request_deserializer=inference__pb2.Session.FromString,
          response_serializer=inference__pb2.Empty.SerializeToString,
      ),
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=inference__pb2.PredictRequest.FromString,
          response_serializer=inference__pb2.PredictResponse.SerializeToString,
      ),
      'GetLogs': grpc.unary_stream_rpc_method_handler(
          servicer.GetLogs,
          request_deserializer=inference__pb2.Empty.FromString,
          response_serializer=inference__pb2.LogEntry.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Inference', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))