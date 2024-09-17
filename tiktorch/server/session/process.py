import multiprocessing as _mp
import pathlib
import tempfile
import uuid
from concurrent.futures import Future
from multiprocessing.connection import Connection
from typing import List, Optional, Tuple, Union

from bioimageio.core import PredictionPipeline, Tensor, create_prediction_pipeline
from bioimageio.spec import load_description
from bioimageio.spec.model import v0_5
from bioimageio.spec.model.v0_5 import BatchAxis

from tiktorch import log
from tiktorch.rpc import Shutdown
from tiktorch.rpc import mp as _mp_rpc
from tiktorch.rpc.mp import BioModelClient, MPServer

from ...converters import Sample
from .backend import base
from .rpc_interface import IRPCModelSession


class SampleValidator:
    def __init__(self, specs: Union[List[v0_5.InputTensorDescr], List[v0_5.OutputTensorDescr]]):
        self._specs = specs

    def check_tensors(self, sample: Sample):
        for tensor_id, tensor_data in sample.members.items():
            self._check_shape(tensor_id, tensor_data)

    def _check_shape(self, tensor_id: str, tensor: Tensor):
        spec = self._get_spec(tensor_id)
        dims_spec = tuple(axis.id for axis in spec.axes)
        if dims_spec != tensor.dims:
            raise ValueError(f"Incompatible axes names, got {tensor.dims} expected {dims_spec}")
        for axis in spec.axes:
            source_axis_size = axis.size
            target_axis_size = tensor.sizes[axis.id]
            if axis.id not in tensor.sizes:
                ValueError(f"{axis.id} not found in {tensor.sizes}")
            if isinstance(axis, BatchAxis) and axis.size is None:
                continue
            if not self._is_size_valid(source_axis_size, target_axis_size):
                raise ValueError(
                    f"Incompatible axis for axis {axis.id} with {source_axis_size}. Got {target_axis_size}"
                )

    def _is_size_valid(self, source_size: Union[int, v0_5.ParameterizedSize, v0_5.SizeReference], target_size: int):
        if isinstance(source_size, v0_5.SizeReference):
            source_size = self._realize_size_reference(source_size)

        if isinstance(source_size, int):
            return source_size == target_size
        elif isinstance(source_size, v0_5.ParameterizedSize):
            min_size = source_size.min
            step_size = source_size.step
            if target_size < min_size:
                return False
            if step_size == 0:
                return min_size == target_size
            diff = target_size - min_size
            num_increments = diff / step_size
            return num_increments % 1 == 0
        else:
            raise ValueError(f"Unexpected size {source_size}")

    def _realize_size_reference(self, size: v0_5.SizeReference) -> Union[int, v0_5.ParameterizedSize]:
        visited = {}

        def dfs(tensor_id, axis_id):
            """
            If size references to another reference and so on, we need to recursively found the ground truth
            """
            if tensor_id in visited:
                return
            visited[tensor_id] = True
            ref_tensor = self._get_spec(tensor_id)
            ref_axes = [axis for axis in ref_tensor.axes if axis.id == axis_id]
            assert len(ref_axes) == 1
            ref_axis = ref_axes[0]
            ref_size = ref_axis.size
            if not isinstance(ref_size, v0_5.SizeReference):
                return ref_size
            return dfs(ref_size.tensor_id, ref_size.axis_id)

        ground_size = dfs(size.tensor_id, size.axis_id)
        if ground_size is None:
            raise ValueError(f"Couldn't realize size reference {size}")
        return ground_size

    def _get_spec(self, tensor_id: str) -> v0_5.InputTensorDescr:
        specs = [spec for spec in self._specs if tensor_id == spec.id]
        if len(specs) == 0:
            raise ValueError(f"Spec {tensor_id} doesn't exist for specs {[spec.id for spec in self._specs]}")
        assert len(specs) == 1, "ids of tensor specs should be unique"
        return specs[0]


class ModelSessionProcess(IRPCModelSession[PredictionPipeline]):
    def __init__(self, model: PredictionPipeline) -> None:
        super().__init__(model)
        self._datasets = {}
        self._worker = base.SessionBackend(self._model)

    def forward(self, sample: Sample) -> Future:
        res = self._worker.forward(sample)
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
) -> Tuple[_mp.Process, BioModelClient]:
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
    api = _mp_rpc.create_client_api(iface_cls=IRPCModelSession, conn=client_conn)
    return proc, BioModelClient(
        input_specs=prediction_pipeline.model_description.inputs,
        output_specs=prediction_pipeline.model_description.outputs,
        api=api,
    )


def _get_prediction_pipeline_from_model_bytes(model_zip: bytes, devices: List[str]) -> PredictionPipeline:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as _tmp_file:
        _tmp_file.write(model_zip)
        temp_file_path = pathlib.Path(_tmp_file.name)
    model = load_description(temp_file_path)
    return create_prediction_pipeline(bioimageio_model=model, devices=devices)
