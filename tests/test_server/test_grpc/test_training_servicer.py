import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import grpc
import h5py
import numpy as np
import pytest
import xarray as xr

from tiktorch.converters import pb_state_to_trainer, pb_tensor_to_xarray, trainer_state_to_pb, xarray_to_pb_tensor
from tiktorch.proto import training_pb2, training_pb2_grpc, utils_pb2
from tiktorch.server.device_pool import TorchDevicePool
from tiktorch.server.grpc import training_servicer
from tiktorch.server.session.backend.base import TrainerSessionBackend
from tiktorch.server.session.process import TrainerSessionProcess
from tiktorch.server.session_manager import SessionManager
from tiktorch.trainer import BaseCallbacks, ShouldStopCallbacks, Trainer, TrainerState


@pytest.fixture(scope="module")
def grpc_add_to_server():
    return training_pb2_grpc.add_TrainingServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer():
    return training_servicer.TrainingServicer(TorchDevicePool(), SessionManager())


@pytest.fixture(autouse=True)
def clean(grpc_servicer):
    yield
    grpc_servicer.close_all_sessions()


@pytest.fixture(scope="module")
def grpc_stub_cls():
    return training_pb2_grpc.TrainingStub


def unet2d_config_path(
    checkpoint_dir: Path, train_data_dir: str, val_data_path: str, resume: Optional[str] = None, device: str = "cpu"
):
    # todo: upsampling makes model torchscript incompatible
    base = f"""
device: {device}  # Use CPU for faster test execution, change to 'cuda' if GPU is available and necessary
model:
  name: UNet2D
  in_channels: 3
  out_channels: 2
  layer_order: gcr
  f_maps: 16
  num_groups: 4
  final_sigmoid: false
  is_segmentation: true
  upsample: default
trainer:
  checkpoint_dir: {checkpoint_dir}
  resume: null
  validate_after_iters: 1
  log_after_iters: 2
  max_num_epochs: 10000
  max_num_iterations: 100000
  eval_score_higher_is_better: True
optimizer:
  learning_rate: 0.0002
  weight_decay: 0.00001
loss:
  name: CrossEntropyLoss
eval_metric:
  name: MeanIoU
  ignore_index: null
lr_scheduler:
  name: MultiStepLR
  milestones: [2, 3]
  gamma: 0.5
loaders:
  dataset: StandardHDF5Dataset
  batch_size: 1
  num_workers: 1
  raw_internal_path: raw
  label_internal_path: label
  weight_internal_path: null
  train:
    file_paths:
      - {train_data_dir}

    slice_builder:
      name: SliceBuilder
      patch_shape: [1, 64, 64]
      stride_shape: [1, 64, 64]
      skip_shape_check: true

    transformer:
      raw:
        - name: Standardize
        - name: RandomFlip
        - name: RandomRotate90
        - name: RandomRotate
          axes: [[2, 1]]
          angle_spectrum: 30
          mode: reflect
        - name: ElasticDeformation
          execution_probability: 1.0
          spline_order: 3
        - name: AdditiveGaussianNoise
          execution_probability: 1.0
        - name: AdditivePoissonNoise
          execution_probability: 1.0
        - name: ToTensor
          expand_dims: true
      label:
        - name: RandomFlip
        - name: RandomRotate90
        - name: RandomRotate
          axes: [[2, 1]]
          angle_spectrum: 30
          mode: reflect
        - name: ElasticDeformation
          execution_probability: 1.0
          spline_order: 0
        - name: ToTensor
          # do not expand dims for cross-entropy loss
          expand_dims: false
          # cross-entropy loss requires target to be of type 'long'
          dtype: 'long'
      weight:
        - name: ToTensor
          expand_dims: false
  val:
    file_paths:
      - {val_data_path}

    slice_builder:
      name: SliceBuilder
      patch_shape: [1, 64, 64]
      stride_shape: [1, 64, 64]
      skip_shape_check: true

    transformer:
      raw:
        - name: Standardize
        - name: ToTensor
          expand_dims: true
      label:
        - name: ToTensor
          expand_dims: false
          dtype: 'long'
      weight:
        - name: ToTensor
          expand_dims: false
"""
    if resume:
        return f"resume: {resume}{base}"
    return base


def create_random_dataset(shape, channel_per_class):
    tmp = tempfile.NamedTemporaryFile(delete=False)

    with h5py.File(tmp.name, "w") as f:
        l_shape = w_shape = shape
        # make sure that label and weight tensors are 3D
        if len(shape) == 4:
            l_shape = shape[1:]
            w_shape = shape[1:]

        if channel_per_class:
            l_shape = (2,) + l_shape

        f.create_dataset("raw", data=np.random.rand(*shape), dtype=np.float32)
        f.create_dataset("label", data=np.random.randint(0, 2, l_shape), dtype=np.int64)
        f.create_dataset("weight_map", data=np.random.rand(*w_shape), dtype=np.float32)

    return tmp.name


def prepare_unet2d_test_environment(resume: Optional[str] = None, device: str = "cpu") -> str:
    checkpoint_dir = Path(tempfile.mkdtemp())

    in_channel = 3
    z = 1  # 2d
    y = 128
    x = 128
    shape = (in_channel, z, y, x)
    binary_loss = False
    train = create_random_dataset(shape, binary_loss)
    val = create_random_dataset(shape, binary_loss)

    config = unet2d_config_path(
        resume=resume, checkpoint_dir=checkpoint_dir, train_data_dir=train, val_data_path=val, device=device
    )
    return config


class TestTrainingServicer:
    def assert_state(self, grpc_stub, training_session_id: str, state_to_check: TrainerState):
        response = grpc_stub.GetStatus(training_session_id)
        assert response.state == trainer_state_to_pb[state_to_check]

    def poll_for_state_grpc(self, grpc_stub, session_id, expected_state: TrainerState, timeout=3, poll_interval=0.1):
        def get_status(*args):
            return pb_state_to_trainer[grpc_stub.GetStatus(session_id).state]

        self.poll_for_state(get_status, expected_state, timeout, poll_interval)

    def poll_for_state(self, get_status: Callable, expected_state: TrainerState, timeout=3, poll_interval=0.1):
        start_time = time.time()

        while True:
            current_state = get_status()

            if current_state == expected_state:
                return current_state

            if time.time() - start_time > timeout:
                pytest.fail(f"Timeout: State did not transition to {expected_state} within {timeout} seconds.")

            time.sleep(poll_interval)

    def test_init_success(self, grpc_stub):
        """
        Test that a session initializes successfully with valid YAML.
        """
        response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        assert response.id is not None, "Failed to initialize training session"

    def test_init_invalid_yaml(self, grpc_stub):
        """
        Test that initializing with invalid YAML raises an error.
        """
        invalid_yaml = "invalid_yaml_content: {unbalanced_braces"
        with pytest.raises(grpc.RpcError) as excinfo:
            grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=invalid_yaml))
        assert "expected ',' or '}', but got" in excinfo.value.details()

    def test_init_failed_then_devices_are_released(self, grpc_stub):
        invalid_yaml = """
        device: cpu
        unknown: 42
        """
        with pytest.raises(grpc.RpcError):
            grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=invalid_yaml))

        # attempt to init with the same device
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        response = utils_pb2.ModelSession(id=init_response.id)
        assert response.id is not None

    def test_start_training_success(self):
        """
        Test starting training after successful initialization.
        """
        trainer_is_called = threading.Event()

        class MockedNominalTrainer(Trainer):
            def __init__(self):
                self.num_epochs = 0
                self.max_num_epochs = 10
                self.num_iterations = 0
                self.max_num_iterations = 100
                self.should_stop_callbacks = ShouldStopCallbacks()
                self.ping_is_best_callbacks = BaseCallbacks()

            def fit(self):
                print("Training has started")
                trainer_is_called.set()

        class MockedTrainerSessionBackend(TrainerSessionProcess):
            def init(self, trainer_yaml_config: str = ""):
                self._worker = TrainerSessionBackend(MockedNominalTrainer())

        backend = MockedTrainerSessionBackend()
        backend.init()
        backend.start_training()
        trainer_is_called.wait(timeout=5)
        backend.shutdown()

    def test_concurrent_state_transitions(self, grpc_stub):
        """
        Test concurrent calls to Start, Pause, and Resume to ensure no deadlocks or race conditions.

        The test should exit gracefully without hanging processes or threads.
        """
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=lambda: grpc_stub.Start(training_session_id)))
            threads.append(threading.Thread(target=lambda: grpc_stub.Pause(training_session_id)))
            threads.append(threading.Thread(target=lambda: grpc_stub.Resume(training_session_id)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def test_queueing_multiple_commands(self, grpc_stub):
        def assert_state(state_to_check):
            self.assert_state(grpc_stub, training_session_id, state_to_check)

        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        grpc_stub.Start(training_session_id)
        assert_state(TrainerState.RUNNING)

        for _ in range(3):
            grpc_stub.Pause(training_session_id)
            assert_state(TrainerState.PAUSED)

            grpc_stub.Resume(training_session_id)
            assert_state(TrainerState.RUNNING)

    def test_error_handling_on_invalid_state_transitions_after_training_started(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        # Attempt to start again while already running
        grpc_stub.Start(training_session_id)
        with pytest.raises(grpc.RpcError) as excinfo:
            grpc_stub.Start(training_session_id)
        assert "Invalid state transition: TrainerState.RUNNING -> TrainerState.RUNNING" in excinfo.value.details()

        # Attempt to pause again while already paused
        grpc_stub.Pause(training_session_id)
        with pytest.raises(grpc.RpcError) as excinfo:
            grpc_stub.Pause(training_session_id)
        assert "Invalid state transition: TrainerState.PAUSED -> TrainerState.PAUSED" in excinfo.value.details()

        # Attempt to resume again while already resumed
        grpc_stub.Resume(training_session_id)
        with pytest.raises(grpc.RpcError) as excinfo:
            grpc_stub.Resume(training_session_id)
        assert "Invalid state transition: TrainerState.RUNNING -> TrainerState.RUNNING" in excinfo.value.details()

    def test_error_handling_on_invalid_state_transitions_before_training_started(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        # Attempt to resume before start
        with pytest.raises(grpc.RpcError) as excinfo:
            grpc_stub.Resume(training_session_id)
        assert "Invalid state transition: TrainerState.IDLE -> TrainerState.RUNNING" in excinfo.value.details()

        # Attempt to pause before start
        with pytest.raises(grpc.RpcError) as excinfo:
            grpc_stub.Pause(training_session_id)
        assert "Invalid state transition: TrainerState.IDLE -> TrainerState.PAUSED" in excinfo.value.details()

    def test_start_training_without_init(self, grpc_stub):
        """
        Test starting training without initializing a session.
        """
        with pytest.raises(grpc.RpcError) as excinfo:
            grpc_stub.Start(utils_pb2.Empty())
        assert excinfo.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "model-session with id  doesn't exist" in excinfo.value.details()

    def test_recover_training_failed(self):
        class MockedExceptionTrainer:
            def __init__(self):
                self.should_stop_callbacks = ShouldStopCallbacks()
                self.ping_is_best_callbacks = BaseCallbacks()

            def fit(self):
                raise Exception("mocked exception")

        class MockedNominalTrainer:
            def __init__(self):
                self.num_epochs = 0
                self.max_num_epochs = 10
                self.num_iterations = 0
                self.max_num_iterations = 100
                self.should_stop_callbacks = ShouldStopCallbacks()
                self.ping_is_best_callbacks = BaseCallbacks()

            def fit(self):
                for epoch in range(self.max_num_epochs):
                    self.num_epochs += 1

        class MockedTrainerSessionBackend(TrainerSessionProcess):
            def init(self, trainer_yaml_config: str):
                if trainer_yaml_config == "nominal":
                    self._worker = TrainerSessionBackend(MockedNominalTrainer())
                elif trainer_yaml_config == "exception":
                    self._worker = TrainerSessionBackend(MockedExceptionTrainer())
                else:
                    # simulate user creating model that raises an exception,
                    # and then adjusts the config for a nominal run
                    raise AssertionError

        backend = MockedTrainerSessionBackend()
        backend.init("exception")
        backend.start_training()

        # client detects that state is failed, closes the session and starts a new one
        self.poll_for_state(backend.get_state, expected_state=TrainerState.FAILED)

        backend.shutdown()

        backend.init("nominal")
        backend.start_training()
        self.poll_for_state(backend.get_state, expected_state=TrainerState.FINISHED)
        backend.shutdown()

    def test_perform_operations_after_training_failed(self):
        def assert_error(func, expected_message: str):
            with pytest.raises(Exception) as excinfo:
                func()
            assert expected_message in str(excinfo.value)

        class MockedExceptionTrainer:
            def __init__(self):
                self.should_stop_callbacks = ShouldStopCallbacks()
                self.ping_is_best_callbacks = BaseCallbacks()

            def fit(self):
                raise Exception("mocked exception")

        class MockedTrainerSessionBackend(TrainerSessionProcess):
            def init(self, trainer_yaml_config: str = ""):
                self._worker = TrainerSessionBackend(MockedExceptionTrainer())

        backend = MockedTrainerSessionBackend()
        backend.init()
        backend.start_training()

        start_thread = threading.Thread(target=backend.start_training)
        start_thread.start()

        pause_thread = threading.Thread(
            target=lambda: assert_error(
                backend.pause_training,
                "Invalid state transition: TrainerState.FAILED -> TrainerState.PAUSED",
            )
        )
        pause_thread.start()

        resume_thread = threading.Thread(
            target=lambda: assert_error(
                backend.pause_training,
                "Invalid state transition: TrainerState.FAILED -> TrainerState.RUNNING",
            )
        )
        resume_thread.start()

        start_thread.join()
        pause_thread.join()
        resume_thread.join()
        backend.shutdown()

    def test_graceful_shutdown_after_init(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)
        grpc_stub.CloseTrainerSession(training_session_id)

    def test_graceful_shutdown_after_start(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)
        grpc_stub.Start(training_session_id)
        grpc_stub.CloseTrainerSession(training_session_id)

    def test_graceful_shutdown_after_pause(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)
        grpc_stub.Start(training_session_id)
        grpc_stub.Pause(training_session_id)
        grpc_stub.CloseTrainerSession(training_session_id)

    def test_graceful_shutdown_after_resume(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)
        grpc_stub.Start(training_session_id)
        grpc_stub.Pause(training_session_id)
        grpc_stub.Resume(training_session_id)
        grpc_stub.CloseTrainerSession(training_session_id)

    def test_close_trainer_session_twice(self, grpc_stub):
        # Attempt to close the session twice
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)
        grpc_stub.CloseTrainerSession(training_session_id)

        # The second attempt should raise an error
        with pytest.raises(grpc.RpcError) as excinfo:
            grpc_stub.CloseTrainerSession(training_session_id)
        assert "Unknown session" in excinfo.value.details()

    def test_forward_while_running(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        grpc_stub.Start(training_session_id)

        batch = 5
        in_channels_unet2d = 3
        out_channels_unet2d = 2
        shape = (batch, in_channels_unet2d, 1, 128, 128)
        data = np.random.rand(*shape).astype(np.float32)
        xarray_data = xr.DataArray(data, dims=("b", "c", "z", "y", "x"))
        pb_tensor = xarray_to_pb_tensor(tensor_id="", array=xarray_data)
        predict_request = utils_pb2.PredictRequest(modelSessionId=training_session_id, tensors=[pb_tensor])

        response = grpc_stub.Predict(predict_request)

        # assert that predict command has retained the init state (e.g. RUNNING)
        self.assert_state(grpc_stub, training_session_id, TrainerState.RUNNING)

        predicted_tensors = [pb_tensor_to_xarray(pb_tensor) for pb_tensor in response.tensors]
        assert len(predicted_tensors) == 1
        predicted_tensor = predicted_tensors[0]
        assert predicted_tensor.dims == ("b", "c", "z", "y", "x")
        assert predicted_tensor.shape == (batch, out_channels_unet2d, 1, 128, 128)

    def test_forward_while_paused(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        grpc_stub.Start(training_session_id)

        batch = 5
        in_channels_unet2d = 3
        out_channels_unet2d = 2
        shape = (batch, in_channels_unet2d, 1, 128, 128)
        data = np.random.rand(*shape).astype(np.float32)
        xarray_data = xr.DataArray(data, dims=("b", "c", "z", "y", "x"))
        pb_tensor = xarray_to_pb_tensor(tensor_id="", array=xarray_data)
        predict_request = utils_pb2.PredictRequest(modelSessionId=training_session_id, tensors=[pb_tensor])

        grpc_stub.Pause(training_session_id)

        response = grpc_stub.Predict(predict_request)

        # assert that predict command has retained the init state (e.g. PAUSED)
        self.assert_state(grpc_stub, training_session_id, TrainerState.PAUSED)

        predicted_tensors = [pb_tensor_to_xarray(pb_tensor) for pb_tensor in response.tensors]
        assert len(predicted_tensors) == 1
        predicted_tensor = predicted_tensors[0]
        assert predicted_tensor.dims == ("b", "c", "z", "y", "x")
        assert predicted_tensor.shape == (batch, out_channels_unet2d, 1, 128, 128)

    def test_save_while_running(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        grpc_stub.Start(training_session_id)

        with tempfile.TemporaryDirectory() as model_checkpoint_dir:
            model_checkpoint_file = Path(model_checkpoint_dir) / "model.pth"
            save_request = training_pb2.SaveRequest(
                modelSessionId=training_session_id, filePath=str(model_checkpoint_file)
            )
            grpc_stub.Save(save_request)
            assert model_checkpoint_file.exists()
            self.assert_state(grpc_stub, training_session_id, TrainerState.RUNNING)

            # assume stopping training to release devices
            grpc_stub.CloseTrainerSession(training_session_id)

            # attempt to init a new model with the new checkpoint and start training
            init_response = grpc_stub.Init(
                training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment(resume=model_checkpoint_file))
            )
            training_session_id = utils_pb2.ModelSession(id=init_response.id)
            grpc_stub.Start(training_session_id)

    def test_save_while_paused(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        grpc_stub.Start(training_session_id)
        time.sleep(1)
        grpc_stub.Pause(training_session_id)

        with tempfile.TemporaryDirectory() as model_checkpoint_dir:
            model_checkpoint_file = Path(model_checkpoint_dir) / "model.pth"
            save_request = training_pb2.SaveRequest(
                modelSessionId=training_session_id, filePath=str(model_checkpoint_file)
            )
            grpc_stub.Save(save_request)
            assert model_checkpoint_file.exists()
            self.assert_state(grpc_stub, training_session_id, TrainerState.PAUSED)

            # assume stopping training to release devices
            grpc_stub.CloseTrainerSession(training_session_id)

            # attempt to init a new model with the new checkpoint and start training
            init_response = grpc_stub.Init(
                training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment(resume=model_checkpoint_file))
            )
            training_session_id = utils_pb2.ModelSession(id=init_response.id)
            grpc_stub.Start(training_session_id)

    def test_export_while_running(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        grpc_stub.Start(training_session_id)

        with tempfile.TemporaryDirectory() as model_checkpoint_dir:
            model_export_file = Path(model_checkpoint_dir) / "bioimageio.zip"
            export_request = training_pb2.ExportRequest(
                modelSessionId=training_session_id, filePath=str(model_export_file)
            )
            grpc_stub.Export(export_request)
            assert model_export_file.exists()
            self.assert_state(grpc_stub, training_session_id, TrainerState.PAUSED)

            # assume stopping training since model is exported
            grpc_stub.CloseTrainerSession(training_session_id)

    def test_export_while_paused(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        grpc_stub.Start(training_session_id)
        time.sleep(1)
        grpc_stub.Pause(training_session_id)

        with tempfile.TemporaryDirectory() as model_checkpoint_dir:
            model_export_file = Path(model_checkpoint_dir) / "bioimageio.zip"
            export_request = training_pb2.ExportRequest(
                modelSessionId=training_session_id, filePath=str(model_export_file)
            )
            grpc_stub.Export(export_request)
            assert model_export_file.exists()
            self.assert_state(grpc_stub, training_session_id, TrainerState.PAUSED)

            # assume stopping training since model is exported
            grpc_stub.CloseTrainerSession(training_session_id)

    def test_best_model_ping(self, grpc_stub):
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)

        grpc_stub.Start(training_session_id)

        responses = grpc_stub.IsBestModel(training_session_id)
        received_updates = 0
        for response in responses:
            assert isinstance(response, utils_pb2.Empty)
            received_updates += 1

            if received_updates >= 3:
                break

    def test_close_session(self, grpc_stub):
        """
        Test closing a training session.
        """
        init_response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        training_session_id = utils_pb2.ModelSession(id=init_response.id)
        grpc_stub.CloseTrainerSession(training_session_id)

        # attempt to perform an operation while session is closed
        operations = [grpc_stub.Start, grpc_stub.Pause, grpc_stub.Resume]
        for operation in operations:
            with pytest.raises(grpc.RpcError) as excinfo:
                operation(training_session_id)
            assert "doesn't exist" in excinfo.value.details()

    def test_multiple_sessions(self, grpc_stub):
        response = grpc_stub.Init(training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment()))
        assert response.id is not None

        response = grpc_stub.Init(
            training_pb2.TrainingConfig(yaml_content=prepare_unet2d_test_environment(device="gpu"))
        )
        assert response.id is not None
