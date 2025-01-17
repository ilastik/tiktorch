# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import utils_pb2 as utils__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\x12\tinference\x1a\x0butils.proto\"W\n\x1f\x43reateDatasetDescriptionRequest\x12\x16\n\x0emodelSessionId\x18\x01 \x01(\t\x12\x0c\n\x04mean\x18\x03 \x01(\x01\x12\x0e\n\x06stddev\x18\x04 \x01(\x01\" \n\x12\x44\x61tasetDescription\x12\n\n\x02id\x18\x01 \x01(\t\"\'\n\x04\x42lob\x12\x0e\n\x06\x66ormat\x18\x01 \x01(\t\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\x0c\"s\n\x19\x43reateModelSessionRequest\x12\x13\n\tmodel_uri\x18\x01 \x01(\tH\x00\x12%\n\nmodel_blob\x18\x02 \x01(\x0b\x32\x0f.inference.BlobH\x00\x12\x11\n\tdeviceIds\x18\x05 \x03(\tB\x07\n\x05model\")\n\tNamedInts\x12\x1c\n\tnamedInts\x18\x01 \x03(\x0b\x32\t.NamedInt\"/\n\x0bNamedFloats\x12 \n\x0bnamedFloats\x18\x01 \x03(\x0b\x32\x0b.NamedFloat\"\x1a\n\x0cModelSession\x12\n\n\x02id\x18\x01 \x01(\t\"\xa8\x01\n\x08LogEntry\x12\x11\n\ttimestamp\x18\x01 \x01(\r\x12(\n\x05level\x18\x02 \x01(\x0e\x32\x19.inference.LogEntry.Level\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\"N\n\x05Level\x12\n\n\x06NOTSET\x10\x00\x12\t\n\x05\x44\x45\x42UG\x10\x01\x12\x08\n\x04INFO\x10\x02\x12\x0b\n\x07WARNING\x10\x03\x12\t\n\x05\x45RROR\x10\x04\x12\x0c\n\x08\x43RITICAL\x10\x05\"U\n\x0ePredictRequest\x12\x16\n\x0emodelSessionId\x18\x01 \x01(\t\x12\x11\n\tdatasetId\x18\x02 \x01(\t\x12\x18\n\x07tensors\x18\x03 \x03(\x0b\x32\x07.Tensor\"+\n\x0fPredictResponse\x12\x18\n\x07tensors\x18\x01 \x03(\x0b\x32\x07.Tensor2\x96\x03\n\tInference\x12U\n\x12\x43reateModelSession\x12$.inference.CreateModelSessionRequest\x1a\x17.inference.ModelSession\"\x00\x12\x36\n\x11\x43loseModelSession\x12\x17.inference.ModelSession\x1a\x06.Empty\"\x00\x12g\n\x18\x43reateDatasetDescription\x12*.inference.CreateDatasetDescriptionRequest\x1a\x1d.inference.DatasetDescription\"\x00\x12*\n\x07GetLogs\x12\x06.Empty\x1a\x13.inference.LogEntry\"\x00\x30\x01\x12!\n\x0bListDevices\x12\x06.Empty\x1a\x08.Devices\"\x00\x12\x42\n\x07Predict\x12\x19.inference.PredictRequest\x1a\x1a.inference.PredictResponse\"\x00\x32G\n\rFlightControl\x12\x18\n\x04Ping\x12\x06.Empty\x1a\x06.Empty\"\x00\x12\x1c\n\x08Shutdown\x12\x06.Empty\x1a\x06.Empty\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _CREATEDATASETDESCRIPTIONREQUEST._serialized_start=43
  _CREATEDATASETDESCRIPTIONREQUEST._serialized_end=130
  _DATASETDESCRIPTION._serialized_start=132
  _DATASETDESCRIPTION._serialized_end=164
  _BLOB._serialized_start=166
  _BLOB._serialized_end=205
  _CREATEMODELSESSIONREQUEST._serialized_start=207
  _CREATEMODELSESSIONREQUEST._serialized_end=322
  _NAMEDINTS._serialized_start=324
  _NAMEDINTS._serialized_end=365
  _NAMEDFLOATS._serialized_start=367
  _NAMEDFLOATS._serialized_end=414
  _MODELSESSION._serialized_start=416
  _MODELSESSION._serialized_end=442
  _LOGENTRY._serialized_start=445
  _LOGENTRY._serialized_end=613
  _LOGENTRY_LEVEL._serialized_start=535
  _LOGENTRY_LEVEL._serialized_end=613
  _PREDICTREQUEST._serialized_start=615
  _PREDICTREQUEST._serialized_end=700
  _PREDICTRESPONSE._serialized_start=702
  _PREDICTRESPONSE._serialized_end=745
  _INFERENCE._serialized_start=748
  _INFERENCE._serialized_end=1154
  _FLIGHTCONTROL._serialized_start=1156
  _FLIGHTCONTROL._serialized_end=1227
# @@protoc_insertion_point(module_scope)
