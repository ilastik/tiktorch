import hashlib

import grpc
import pytest

from tiktorch.server.data_store import DataStore
from tiktorch.server.grpc import data_store_servicer

import data_store_pb2
import data_store_pb2_grpc


@pytest.fixture(scope="module")
def grpc_add_to_server():
    return data_store_pb2_grpc.add_DataStoreServicer_to_server


@pytest.fixture(scope="module")
def data_store():
    return DataStore()


@pytest.fixture(scope="module")
def grpc_servicer(data_store):
    return data_store_servicer.DataStoreServicer(data_store)


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    return data_store_pb2_grpc.DataStoreStub


class TestUpload:
    def test_calling_upload_without_header_raises(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as e:
            content = [data_store_pb2.UploadRequest(content=b"abc")]
            res = grpc_stub.Upload(iter(content))

    def test_calling_upload(self, grpc_stub, data_store):
        content = b"aabbbacaaraa"
        sha256_hash = hashlib.sha256(content).hexdigest()

        def _gen():
            yield data_store_pb2.UploadRequest(info=data_store_pb2.UploadInfo(size=12))
            for i in range(0, len(content), 3):
                yield data_store_pb2.UploadRequest(content=content[i : i + 3])

        res = grpc_stub.Upload(_gen())
        assert res.id
        assert res.size == 12
        assert res.sha256 == sha256_hash

        data = data_store.get(res.id)
        assert data == content
