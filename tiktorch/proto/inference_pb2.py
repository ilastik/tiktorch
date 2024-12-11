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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\x12\tinference\x1a\x0butils.proto\"c\n\x06\x44\x65vice\x12\n\n\x02id\x18\x01 \x01(\t\x12(\n\x06status\x18\x02 \x01(\x0e\x32\x18.inference.Device.Status\"#\n\x06Status\x12\r\n\tAVAILABLE\x10\x00\x12\n\n\x06IN_USE\x10\x01\"W\n\x1f\x43reateDatasetDescriptionRequest\x12\x16\n\x0emodelSessionId\x18\x01 \x01(\t\x12\x0c\n\x04mean\x18\x03 \x01(\x01\x12\x0e\n\x06stddev\x18\x04 \x01(\x01\" \n\x12\x44\x61tasetDescription\x12\n\n\x02id\x18\x01 \x01(\t\"\'\n\x04\x42lob\x12\x0e\n\x06\x66ormat\x18\x01 \x01(\t\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\x0c\"s\n\x19\x43reateModelSessionRequest\x12\x13\n\tmodel_uri\x18\x01 \x01(\tH\x00\x12%\n\nmodel_blob\x18\x02 \x01(\x0b\x32\x0f.inference.BlobH\x00\x12\x11\n\tdeviceIds\x18\x05 \x03(\tB\x07\n\x05model\")\n\tNamedInts\x12\x1c\n\tnamedInts\x18\x01 \x03(\x0b\x32\t.NamedInt\"/\n\x0bNamedFloats\x12 \n\x0bnamedFloats\x18\x01 \x03(\x0b\x32\x0b.NamedFloat\"\x1a\n\x0cModelSession\x12\n\n\x02id\x18\x01 \x01(\t\"\xa8\x01\n\x08LogEntry\x12\x11\n\ttimestamp\x18\x01 \x01(\r\x12(\n\x05level\x18\x02 \x01(\x0e\x32\x19.inference.LogEntry.Level\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\"N\n\x05Level\x12\n\n\x06NOTSET\x10\x00\x12\t\n\x05\x44\x45\x42UG\x10\x01\x12\x08\n\x04INFO\x10\x02\x12\x0b\n\x07WARNING\x10\x03\x12\t\n\x05\x45RROR\x10\x04\x12\x0c\n\x08\x43RITICAL\x10\x05\"-\n\x07\x44\x65vices\x12\"\n\x07\x64\x65vices\x18\x01 \x03(\x0b\x32\x11.inference.Device\"U\n\x0ePredictRequest\x12\x16\n\x0emodelSessionId\x18\x01 \x01(\t\x12\x11\n\tdatasetId\x18\x02 \x01(\t\x12\x18\n\x07tensors\x18\x03 \x03(\x0b\x32\x07.Tensor\"+\n\x0fPredictResponse\x12\x18\n\x07tensors\x18\x01 \x03(\x0b\x32\x07.Tensor\"\x07\n\x05\x45mpty2\xbe\x03\n\tInference\x12U\n\x12\x43reateModelSession\x12$.inference.CreateModelSessionRequest\x1a\x17.inference.ModelSession\"\x00\x12@\n\x11\x43loseModelSession\x12\x17.inference.ModelSession\x1a\x10.inference.Empty\"\x00\x12g\n\x18\x43reateDatasetDescription\x12*.inference.CreateDatasetDescriptionRequest\x1a\x1d.inference.DatasetDescription\"\x00\x12\x34\n\x07GetLogs\x12\x10.inference.Empty\x1a\x13.inference.LogEntry\"\x00\x30\x01\x12\x35\n\x0bListDevices\x12\x10.inference.Empty\x1a\x12.inference.Devices\"\x00\x12\x42\n\x07Predict\x12\x19.inference.PredictRequest\x1a\x1a.inference.PredictResponse\"\x00\x32o\n\rFlightControl\x12,\n\x04Ping\x12\x10.inference.Empty\x1a\x10.inference.Empty\"\x00\x12\x30\n\x08Shutdown\x12\x10.inference.Empty\x1a\x10.inference.Empty\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DEVICE._serialized_start=43
  _DEVICE._serialized_end=142
  _DEVICE_STATUS._serialized_start=107
  _DEVICE_STATUS._serialized_end=142
  _CREATEDATASETDESCRIPTIONREQUEST._serialized_start=144
  _CREATEDATASETDESCRIPTIONREQUEST._serialized_end=231
  _DATASETDESCRIPTION._serialized_start=233
  _DATASETDESCRIPTION._serialized_end=265
  _BLOB._serialized_start=267
  _BLOB._serialized_end=306
  _CREATEMODELSESSIONREQUEST._serialized_start=308
  _CREATEMODELSESSIONREQUEST._serialized_end=423
  _NAMEDINTS._serialized_start=425
  _NAMEDINTS._serialized_end=466
  _NAMEDFLOATS._serialized_start=468
  _NAMEDFLOATS._serialized_end=515
  _MODELSESSION._serialized_start=517
  _MODELSESSION._serialized_end=543
  _LOGENTRY._serialized_start=546
  _LOGENTRY._serialized_end=714
  _LOGENTRY_LEVEL._serialized_start=636
  _LOGENTRY_LEVEL._serialized_end=714
  _DEVICES._serialized_start=716
  _DEVICES._serialized_end=761
  _PREDICTREQUEST._serialized_start=763
  _PREDICTREQUEST._serialized_end=848
  _PREDICTRESPONSE._serialized_start=850
  _PREDICTRESPONSE._serialized_end=893
  _EMPTY._serialized_start=895
  _EMPTY._serialized_end=902
  _INFERENCE._serialized_start=905
  _INFERENCE._serialized_end=1351
  _FLIGHTCONTROL._serialized_start=1353
  _FLIGHTCONTROL._serialized_end=1464
# @@protoc_insertion_point(module_scope)
