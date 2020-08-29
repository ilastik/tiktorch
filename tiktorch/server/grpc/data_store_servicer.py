import time

import grpc
import hashlib

import data_store_pb2
import data_store_pb2_grpc

from tiktorch.server.data_store import IDataStore


class DataStoreServicer(data_store_pb2_grpc.DataStoreServicer):
    def __init__(self, data_store: IDataStore) -> None:
        self.__data_store = data_store

    def Upload(
        self, request_iterator: data_store_pb2.UploadRequest, context
    ) -> data_store_pb2.UploadResponse:
        # TODO: Move hash to data_store

        rq = next(request_iterator)
        if not rq.HasField("info"):
            raise ValueError("Header information is not provided")

        data = b""
        sha256 = hashlib.sha256()
        for rq in request_iterator:
            data += rq.content
            sha256.update(rq.content)

        id_ = self.__data_store.put(data)

        return data_store_pb2.UploadResponse(id=id_, size=len(data), sha256=sha256.hexdigest())
