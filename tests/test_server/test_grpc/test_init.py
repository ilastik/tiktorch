import json
import os
import threading

import grpc

from tiktorch.proto.inference_pb2 import Empty
from tiktorch.proto.inference_pb2_grpc import FlightControlStub
from tiktorch.server.grpc import serve
from tiktorch.utils import wait


def test_serving_on_random_port(tmpdir):
    conn_file_path = str(tmpdir / "conn.json")

    def _server():
        serve("127.0.0.1", 0, connection_file_path=conn_file_path)

    srv_thread = threading.Thread(target=_server)
    srv_thread.start()

    wait(lambda: os.path.exists(conn_file_path))

    with open(conn_file_path, "r") as conn_file:
        conn_data = json.load(conn_file)

    assert conn_data["addr"] == "127.0.0.1"
    assert conn_data["port"] > 0

    addr, port = conn_data["addr"], conn_data["port"]

    chan = grpc.insecure_channel(f"{addr}:{port}")
    client = FlightControlStub(chan)

    result = client.Ping(Empty())
    assert isinstance(result, Empty)
    client.Shutdown(Empty())
