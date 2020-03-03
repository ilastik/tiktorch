import grpc

from tiktorch._generated import inference_pb2, inference_pb2_grpc


def run():
    with grpc.insecure_channel("127.0.0.1:5567") as channel:
        stub = inference_pb2_grpc.InferenceStub(channel)
        response = stub.ListDevices(inference_pb2.Empty())
        print(response)


if __name__ == "__main__":
    run()
