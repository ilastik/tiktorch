import multiprocessing as _mp
import pathlib
import tempfile
import uuid
from concurrent.futures import Future
from multiprocessing.connection import Connection
from typing import List, Optional, Tuple, Type, TypeVar, Union

from bioimageio.core import PredictionPipeline, Tensor, create_prediction_pipeline
from bioimageio.spec import InvalidDescr, load_description
from bioimageio.spec.model import v0_5

from tiktorch import log
from tiktorch.rpc import Shutdown
from tiktorch.rpc import mp as _mp_rpc
from tiktorch.rpc.interface import RPCInterface
from tiktorch.rpc.mp import BioModelClient, MPServer

from ...converters import Sample
from ...trainer import TrainerYamlParser
from .backend import base
from .rpc_interface import IRPCModelSession, IRPCTrainer


class InputSampleValidator:
    def __init__(self, specs: List[v0_5.InputTensorDescr]):
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
            if source_axis_size is None:
                continue
            try:
                self._validate_size(source_axis_size, target_axis_size)
            except ValueError as e:
                raise ValueError(f"Incompatible axis for axis {axis.id}. Reason: {e} ")

    def _validate_size(self, source_size: Union[int, v0_5.ParameterizedSize, v0_5.SizeReference], target_size: int):
        if isinstance(source_size, v0_5.SizeReference):
            source_size = self._realize_size_reference(source_size)

        if isinstance(source_size, int):
            if source_size != target_size:
                raise ValueError(f"{source_size} != {target_size}")
        elif isinstance(source_size, v0_5.ParameterizedSize):
            source_size.validate_size(target_size)
        else:
            raise ValueError(f"Unexpected size {source_size}")

    def _realize_size_reference(self, size: v0_5.SizeReference) -> Union[int, v0_5.ParameterizedSize]:
        ref_tensor = self._get_spec(size.tensor_id)
        ref_axes = [axis for axis in ref_tensor.axes if axis.id == size.axis_id]
        assert len(ref_axes) == 1
        ref_axis = ref_axes[0]
        return ref_axis.size

    def _get_spec(self, tensor_id: str) -> v0_5.InputTensorDescr:
        specs = [spec for spec in self._specs if tensor_id == spec.id]
        if len(specs) == 0:
            raise ValueError(
                f"Spec '{tensor_id}' doesn't exist for specs {','.join([spec.id for spec in self._specs])}"
            )
        assert len(specs) == 1, "ids of tensor specs should be unique"
        return specs[0]


class ModelSessionProcess(IRPCModelSession):
    def __init__(self) -> None:
        super().__init__()
        self._datasets = {}
        self._worker: Optional[base.BioModelSessionBackend] = None

    def init(self, model_bytes: bytes, devices: List[str]):
        prediction_pipeline = _get_prediction_pipeline_from_model_bytes(model_bytes, devices)
        self._worker = base.BioModelSessionBackend(prediction_pipeline)

    def forward(self, sample: Sample) -> Future:
        res = self.worker.forward(sample)
        return res

    def create_dataset(self, mean, stddev):
        id_ = uuid.uuid4().hex
        self._datasets[id_] = {"mean": mean, "stddev": stddev}
        return id_

    def shutdown(self) -> Shutdown:
        if self._worker is None:
            return Shutdown()
        self.worker.shutdown()
        return Shutdown()

    @property
    def worker(self) -> base.BioModelSessionBackend:
        if self._worker is None:
            raise ValueError("Server isn't initialized")
        return self._worker


class TrainerSessionProcess(IRPCTrainer):
    def __init__(self):
        self._worker: Optional[base.TrainerSessionBackend] = None

    @property
    def worker(self) -> base.TrainerSessionBackend:
        if self._worker is None:
            raise ValueError("Server isn't initialized")
        return self._worker

    def init(self, trainer_yaml_config):
        parser = TrainerYamlParser(trainer_yaml_config)
        trainer = parser.parse()
        self._worker = base.TrainerSessionBackend(trainer)

    def forward(self, input_tensors) -> Future:
        res = self.worker.forward(input_tensors)
        return res

    def resume_training(self):
        self.worker.resume_training()

    def start_training(self):
        self.worker.start_training()

    def pause_training(self):
        self.worker.pause_training()

    def save(self):
        self.worker.save()

    def export(self):
        self.worker.export()

    def get_state(self):
        return self.worker.get_state()

    def shutdown(self):
        if self._worker is None:
            return Shutdown()
        self.worker.shutdown()
        return Shutdown()


def _run_server(api: RPCInterface, conn: Connection, log_queue: Optional[_mp.Queue] = None):
    try:
        # from: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    except ModuleNotFoundError:
        pass  # probably running on windows

    if log_queue:
        log.configure(log_queue)

    srv = MPServer(api, conn)
    srv.listen()


T = TypeVar("T", bound=RPCInterface)


def start_trainer_process(log_queue: Optional[_mp.Queue] = None) -> Tuple[_mp.Process, TrainerSessionProcess]:
    return start_process(interface_class=TrainerSessionProcess, log_queue=log_queue)


def start_process(interface_class: Type[T], log_queue: Optional[_mp.Queue] = None) -> Tuple[_mp.Process, T]:
    client_conn, server_conn = _mp.Pipe()
    proc = _mp.Process(
        target=_run_server,
        name="TiktorchProcess",
        kwargs={"conn": server_conn, "log_queue": log_queue, "api": interface_class()},
    )
    proc.start()
    api: T = _mp_rpc.create_client_api(iface_cls=interface_class, conn=client_conn)
    return proc, api


def start_model_session_process(
    model_bytes: bytes, log_queue: Optional[_mp.Queue] = None
) -> Tuple[_mp.Process, BioModelClient]:
    model_descr = _get_model_descr_from_model_bytes(model_bytes)
    proc, api = start_process(interface_class=ModelSessionProcess, log_queue=log_queue)
    return proc, BioModelClient(
        input_specs=model_descr.inputs,
        output_specs=model_descr.outputs,
        api=api,
    )


def _get_prediction_pipeline_from_model_bytes(model_bytes: bytes, devices: List[str]) -> PredictionPipeline:
    return create_prediction_pipeline(bioimageio_model=_get_model_descr_from_model_bytes(model_bytes), devices=devices)


def _get_model_descr_from_model_bytes(model_bytes: bytes) -> v0_5.ModelDescr:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as _tmp_file:
        _tmp_file.write(model_bytes)
        temp_file_path = pathlib.Path(_tmp_file.name)
    model_descr = load_description(temp_file_path, format_version="latest")
    if isinstance(model_descr, InvalidDescr):
        raise ValueError(f"Failed to load valid model descriptor {model_descr.validation_summary}")
    return model_descr
