import argparse
import os

import grpc
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.io import imread

from tiktorch import converters
from tiktorch.proto import training_pb2, training_pb2_grpc, utils_pb2


def reorder_axes(input_arr: np.ndarray, *, from_axes_tags: str, to_axes_tags: str) -> np.ndarray:
    tagged_input = xr.DataArray(input_arr, dims=tuple(from_axes_tags))

    axes_removed = set(from_axes_tags).difference(to_axes_tags)
    axes_added = set(to_axes_tags).difference(from_axes_tags)

    output = tagged_input.squeeze(tuple(axes_removed)).expand_dims(tuple(axes_added)).transpose(*tuple(to_axes_tags))
    assert len(output.shape) == len(to_axes_tags)
    return output.data.astype("float32")


def expand_loaders_path(yaml_path) -> str:
    with open(yaml_path, "r") as f:
        config = f.read()
    yaml_config = yaml.safe_load(config)
    train_files_path = yaml_config["loaders"]["train"]["file_paths"]
    assert len(train_files_path) == 1, "we assume that it is a directory with all the training subdirectories"
    val_files_path = yaml_config["loaders"]["val"]["file_paths"]
    assert len(val_files_path) == 1, "we assume that it is a directory with all the training subdirectories"
    train_file_path = train_files_path[0]
    val_file_path = val_files_path[0]

    train_files = os.listdir(train_file_path)
    val_files = os.listdir(val_file_path)
    train_files = [os.path.join(train_file_path, f) for f in train_files]
    val_files = [os.path.join(val_file_path, f) for f in val_files]
    yaml_config["loaders"]["train"]["file_paths"] = train_files
    yaml_config["loaders"]["val"]["file_paths"] = val_files

    # convert yaml_config to string
    config = yaml.dump(yaml_config)
    return config


class TrainingClient:
    def __init__(self, host="127.0.0.1", port=5567):
        print("Connecting to server...")
        print(f"Host: {host}, Port: {port}")
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        print(f"Channel: {self.channel}")
        self.stub = training_pb2_grpc.TrainingStub(self.channel)

    def init(self, yaml_path):
        config = expand_loaders_path(yaml_path)
        try:
            response = self.stub.Init(training_pb2.TrainingConfig(yaml_content=config))
            print(f"Training session initialized with ID: {response.id}")
        except grpc.RpcError as e:
            print(f"Error during Init: {e}")

    def start(self, session_id):
        try:
            self.stub.Start(utils_pb2.ModelSession(id=session_id))
            print("Training started.")
        except grpc.RpcError as e:
            print(f"Error during Start: {e}")

    def pause(self, session_id):
        try:
            self.stub.Pause(utils_pb2.ModelSession(id=session_id))
            print("Training paused.")
        except grpc.RpcError as e:
            print(f"Error during Pause: {e}")

    def resume(self, session_id):
        try:
            self.stub.Resume(utils_pb2.ModelSession(id=session_id))
            print("Training resumed.")
        except grpc.RpcError as e:
            print(f"Error during Resume: {e}")

    def forward(self, session_id, image_file_path):
        try:
            # load image
            image = imread(image_file_path)
            print("image shape", image.shape)
            print("min", image.min())
            print("max", image.max())
            reordered_image = reorder_axes(image, from_axes_tags="yx", to_axes_tags="bczyx")
            pb_tensors = converters.numpy_to_pb_tensor("input", reordered_image, axistags="bczyx")

            training_session_id = utils_pb2.ModelSession(id=session_id)
            forward_request = utils_pb2.PredictRequest(modelSessionId=training_session_id, tensors=[pb_tensors])
            server_response = self.stub.Predict(forward_request)
            results = [converters.pb_tensor_to_numpy(t) for t in server_response.tensors]
            results = [reorder_axes(r, from_axes_tags="bczyx", to_axes_tags="yx") for r in results]
            assert len(results) == 1

            result = results[0]
            print("Received result shape", result.shape)
            print("max", result.max())
            print("min", result.min())

            # Create subplots for side-by-side images
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Original image
            im1 = axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")  # Turn off axis labels

            # Add a colorbar for the original image
            divider1 = make_axes_locatable(axes[0])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im1, cax=cax1)

            # Predicted image
            im2 = axes[1].imshow(result, cmap="gray")
            axes[1].set_title("Predicted Image")
            axes[1].axis("off")  # Turn off axis labels

            # Add a colorbar for the predicted image
            divider2 = make_axes_locatable(axes[1])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im2, cax=cax2)

            # Adjust layout
            plt.tight_layout()
            plt.show()
        except grpc.RpcError as e:
            print(f"Error during Forward: {e}")

    def forward_tensor(self, session_id, tensor_file_path):
        try:
            # load tensor
            tensor = torch.load(tensor_file_path).detach().numpy()
            print("tesnor shape", tensor.shape)

            # reordered_image = reorder_axes(image, from_axes_tags="yx", to_axes_tags="bczyx")
            pb_tensors = converters.numpy_to_pb_tensor("input", tensor)

            training_session_id = utils_pb2.ModelSession(id=session_id)
            forward_request = utils_pb2.PredictRequest(modelSessionId=training_session_id, tensors=[pb_tensors])
            server_response = self.stub.Predict(forward_request)
            results = [converters.pb_tensor_to_numpy(t) for t in server_response.tensors]
            results = [reorder_axes(r, from_axes_tags="bczyx", to_axes_tags="yx") for r in results]
            assert len(results) == 1

            result = results[0]
            result = results[0]
            print("max", result.max())
            print("min", result.min())

            plt.imshow(result, cmap="gray")
            plt.colorbar()  # Optional: Display a color bar for intensity values
            plt.title("Grayscale Image")
            plt.axis("off")  # Optional: Turn off axis labels
            plt.show()
            print("Training forwarded.")
        except grpc.RpcError as e:
            print(f"Error during Forward: {e}")

    def save(self, file_path, session_id):
        try:
            training_session_id = utils_pb2.ModelSession(id=session_id)
            save_request = training_pb2.SaveRequest(modelSessionId=training_session_id, filePath=file_path)
            self.stub.Save(save_request)
            print("Training saved.")
        except grpc.RpcError as e:
            print(f"Error during Save: {e}")

    def export(self, file_path, session_id):
        try:
            training_session_id = utils_pb2.ModelSession(id=session_id)
            export_request = training_pb2.ExportRequest(modelSessionId=training_session_id, filePath=file_path)
            self.stub.Export(export_request)
            print("Training exported.")
        except grpc.RpcError as e:
            print(f"Error during Export: {e}")

    def is_best(self, session_id):
        try:
            stream = self.stub.GetBestModelIdx(utils_pb2.ModelSession(id=session_id))
            for i, res in enumerate(stream):
                print(f"Training is best id {res.id}.")
        except grpc.RpcError as e:
            print(f"Error during Export: {e}")

    def get_status(self, session_id):
        try:
            response = self.stub.GetStatus(utils_pb2.ModelSession(id=session_id))
            print(f"Training status: {response.state}")
        except grpc.RpcError as e:
            print(f"Error during GetStatus: {e}")

    def close_session(self, session_id):
        try:
            self.stub.CloseTrainerSession(utils_pb2.ModelSession(id=session_id))
            print("Training session closed.")
        except grpc.RpcError as e:
            print(f"Error during CloseTrainerSession: {e}")


def main():
    parser = argparse.ArgumentParser(description="CLI for Training Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server hostname")
    parser.add_argument("--port", type=int, default=5567, help="Server port")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Init
    init_parser = subparsers.add_parser("init", help="Initialize a training session")
    init_parser.add_argument("--yaml", type=str, required=True, help="YAML configuration for training")

    # Start
    start_parser = subparsers.add_parser("start", help="Start training")
    start_parser.add_argument("--session-id", type=str, required=True, help="Session ID to use")

    # Pause
    pause_parser = subparsers.add_parser("pause", help="Pause training")
    pause_parser.add_argument("--session-id", type=str, required=True, help="Session ID to use")

    # Resume
    resume_parser = subparsers.add_parser("resume", help="Resume training")
    resume_parser.add_argument("--session-id", type=str, required=True, help="Session ID to use")
    # Forward
    forward_parser = subparsers.add_parser("forward", help="Forward the training state to the client")
    forward_parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session ID to use",
    )
    forward_parser.add_argument("--image-file-path", type=str, required=True, help="file path to use")

    # Forward with preprocessed tensor
    forward_parser = subparsers.add_parser("forward-tensor", help="Forward the training state to the client")
    forward_parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session ID to use",
    )
    forward_parser.add_argument("--tensor-file-path", type=str, required=True, help="file path to use")

    # Save
    save_parser = subparsers.add_parser("save", help="Save the training state")
    save_parser.add_argument("--session-id", type=str, required=True, help="Session ID to use")
    save_parser.add_argument("--file-path", type=str, required=True, help="file path to use")

    # Export
    export_parser = subparsers.add_parser("export", help="Export the trained model")
    export_parser.add_argument("--session-id", type=str, required=True, help="Session ID to use")
    export_parser.add_argument("--file-path", type=str, required=True, help="file path to use")

    # Get Status
    status_parser = subparsers.add_parser("status", help="Get the current training status")
    status_parser.add_argument("--session-id", type=str, required=True, help="Session ID to use")

    # Get Status
    is_best_parser = subparsers.add_parser("is_best", help="Best model notification")
    is_best_parser.add_argument("--session-id", type=str, required=True, help="Session ID to use")

    # Close Session
    close_parser = subparsers.add_parser("close", help="Close the training session")
    close_parser.add_argument("--session-id", type=str, required=True, help="Session ID to use")

    args = parser.parse_args()

    # Create a client
    client = TrainingClient(host=args.host, port=args.port)

    # Command execution
    if args.command == "init":
        client.init(args.yaml)
    elif args.command == "start":
        client.start(args.session_id)
    elif args.command == "pause":
        client.pause(args.session_id)
    elif args.command == "resume":
        client.resume(args.session_id)
    elif args.command == "forward":
        client.forward(args.session_id, args.image_file_path)
    elif args.command == "save":
        client.save(args.file_path, args.session_id)
    elif args.command == "export":
        client.export(args.file_path, args.session_id)
    elif args.command == "status":
        client.get_status(args.session_id)
    elif args.command == "close":
        client.close_session(args.session_id)
    elif args.command == "is_best":
        client.is_best(args.session_id)
    elif args.command == "forward_tensor":
        client.forward_tensor(args.session_id, args.tensor_file_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
