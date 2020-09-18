import hashlib
import logging
import time

import grpc

from tiktorch.proto import data_store_pb2, data_store_pb2_grpc
from tiktorch.server.data_store import IDataStore

logger = logging.getLogger(__name__)


class DataStoreServicer(data_store_pb2_grpc.DataStoreServicer):
    def __init__(self, data_store: IDataStore) -> None:
        self.__data_store = data_store

    def Upload(self, request_iterator: data_store_pb2.UploadRequest, context) -> data_store_pb2.UploadResponse:
        # TODO: Move hash to data_store

        rq = next(request_iterator)
        if not rq.HasField("info"):
            raise ValueError("Header information is not provided")

        expected_size = rq.info.size
        data = b""
        sha256 = hashlib.sha256()

        for rq in request_iterator:
            data += rq.content
            sha256.update(rq.content)

        id_ = self.__data_store.put(data)
        if expected_size != len(data):
            logger.debug("Upload truncated expected %s bytes but received only %s", expected_size, len(data))
            raise RuntimeError(f"Expected data of size {expected_size} bytes but got only {len(data)}")

        return data_store_pb2.UploadResponse(id=id_, size=len(data), sha256=sha256.hexdigest())
