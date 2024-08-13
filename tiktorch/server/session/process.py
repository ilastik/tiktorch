import multiprocessing as _mp
import pathlib
import tempfile
import uuid
from concurrent.futures import Future
from multiprocessing.connection import Connection
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from bioimageio.core import load_resource_description
from bioimageio.core.prediction_pipeline import PredictionPipeline, create_prediction_pipeline
from bioimageio.core.resource_io import nodes
from bioimageio.core.resource_io.nodes import ParametrizedInputShape

from tiktorch import log
from tiktorch.rpc import Shutdown
from tiktorch.rpc import mp as _mp_rpc
from tiktorch.rpc.mp import MPServer

from ...converters import Sample
from .backend import base
from .rpc_interface import IRPCModelSession


class InputTensorValidator:
    def __init__(self, model: PredictionPipeline):
        self._model = model

    def check_tensors(self, sample: Sample):
        for tensor_id, tensor in sample.tensors.items():
            self.check_shape(tensor_id, tensor.dims, tensor.shape)

    def _get_input_tensors_with_names(self) -> Dict[str, nodes.InputTensor]:
        return {tensor.name: tensor for tensor in self._model.input_specs}

    def check_shape(self, tensor_id: str, axes: Tuple[str, ...], shape: Tuple[int, ...]):
        shape = self._get_axes_with_size(axes, shape)
        spec = self._get_input_spec(tensor_id)
        if isinstance(spec.shape, list):
            self._check_shape_explicit(spec, shape)
        elif isinstance(spec.shape, ParametrizedInputShape):
            self._check_shape_parameterized(spec, shape)
        else:
            raise ValueError(f"Unexpected shape {spec.shape}")

    def _get_input_spec(self, tensor_id: str) -> nodes.InputTensor:
        self._check_spec_exists(tensor_id)
        specs = [spec for spec in self._model.input_specs if spec.name == tensor_id]
        assert len(specs) == 1, "ids of tensor specs should be unique"
        return specs[0]

    def _check_spec_exists(self, tensor_id: str):
        spec_names = [spec.name for spec in self._model.input_specs]
        if tensor_id not in spec_names:
            raise ValueError(f"Spec {tensor_id} doesn't exist for specs {spec_names}")

    def _check_shape_explicit(self, spec: nodes.InputTensor, tensor_shape: Dict[str, int]):
        assert self._is_shape_explicit(spec)
        reference_shape = {name: size for name, size in zip(spec.axes, spec.shape)}
        self._check_same_axes(reference_shape, tensor_shape)
        if reference_shape != tensor_shape:
            raise ValueError(f"Incompatible shapes found {tensor_shape}, expected {reference_shape}")

    def _check_shape_parameterized(self, spec: nodes.InputTensor, tensor_shape: Dict[str, int]):
        assert isinstance(spec.shape, ParametrizedInputShape)
        if not self._is_shape(tensor_shape.values()):
            raise ValueError(f"Invalid shape's sizes {tensor_shape}")

        min_shape = self._get_axes_with_size(spec.axes, tuple(spec.shape.min))
        step = self._get_axes_with_size(spec.axes, tuple(spec.shape.step))
        self._check_same_axes(tensor_shape, min_shape)

        tensor_shapes_arr = np.array(list(tensor_shape.values()))
        min_shape_arr = np.array(list(min_shape.values()))
        step_arr = np.array(list(step.values()))
        diff = tensor_shapes_arr - min_shape_arr
        if any(size < 0 for size in diff):
            raise ValueError(f"Tensor shape {tensor_shape} smaller than min shape {min_shape}")

        non_zero_idx = np.nonzero(step_arr)
        multipliers = diff[non_zero_idx] / step_arr[non_zero_idx]
        multiplier = np.unique(multipliers)
        if len(multiplier) == 1 and self._is_natural_number(multiplier[0]):
            return
        raise ValueError(f"Tensor shape {tensor_shape} not valid for spec {spec}")

    def _check_same_axes(self, source: Dict[str, int], target: Dict[str, int]):
        if source.keys() != target.keys():
            raise ValueError(f"Incompatible axes for tensor {target} and reference {source}")

    def _is_natural_number(self, n) -> bool:
        return n % 1 == 0.0 and n >= 0

    def _is_shape(self, shape: Iterator[int]) -> bool:
        return all(self._is_natural_number(dim) for dim in shape)

    def _get_axes_with_size(self, axes: Tuple[str, ...], shape: Tuple[int, ...]) -> Dict[str, int]:
        assert len(axes) == len(shape)
        return {name: size for name, size in zip(axes, shape)}

    def _is_shape_explicit(self, spec: nodes.InputTensor) -> bool:
        return isinstance(spec.shape, list)


class ModelSessionProcess(IRPCModelSession[PredictionPipeline]):
    def __init__(self, model: PredictionPipeline) -> None:
        super().__init__(model)
        self._datasets = {}
        self._worker = base.SessionBackend(self._model)
        self._shape_validator = InputTensorValidator(self._model)

    def forward(self, sample: Sample) -> Future:
        tensors_data = [sample.tensors[tensor.name] for tensor in self.model.input_specs]
        res = self._worker.forward(tensors_data)
        return res

    def create_dataset(self, mean, stddev):
        id_ = uuid.uuid4().hex
        self._datasets[id_] = {"mean": mean, "stddev": stddev}
        return id_

    def shutdown(self) -> Shutdown:
        self._worker.shutdown()
        return Shutdown()


def _run_model_session_process(
    conn: Connection, prediction_pipeline: PredictionPipeline, log_queue: Optional[_mp.Queue] = None
):
    try:
        # from: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    except ModuleNotFoundError:
        pass  # probably running on windows

    if log_queue:
        log.configure(log_queue)

    session_proc = ModelSessionProcess(prediction_pipeline)
    srv = MPServer(session_proc, conn)
    srv.listen()


def start_model_session_process(
    model_zip: bytes, devices: List[str], log_queue: Optional[_mp.Queue] = None
) -> Tuple[_mp.Process, IRPCModelSession]:
    client_conn, server_conn = _mp.Pipe()
    prediction_pipeline = _get_prediction_pipeline_from_model_bytes(model_zip, devices)
    proc = _mp.Process(
        target=_run_model_session_process,
        name="ModelSessionProcess",
        kwargs={
            "conn": server_conn,
            "log_queue": log_queue,
            "prediction_pipeline": prediction_pipeline,
        },
    )
    proc.start()
    # here create the prediction pipeline, share it to the model session class and the client
    return proc, _mp_rpc.create_client(
        iface_cls=IRPCModelSession, api_kwargs={"model": prediction_pipeline}, conn=client_conn
    )


def _get_prediction_pipeline_from_model_bytes(model_zip: bytes, devices: List[str]) -> PredictionPipeline:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as _tmp_file:
        _tmp_file.write(model_zip)
        temp_file_path = pathlib.Path(_tmp_file.name)
    model = load_resource_description(temp_file_path)
    return create_prediction_pipeline(bioimageio_model=model, devices=devices)
