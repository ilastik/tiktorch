import multiprocessing as _mp
import pathlib
import tempfile
import uuid
from concurrent.futures import Future
from multiprocessing.connection import Connection
from typing import List, Optional, Tuple

from bioimageio.core import load_resource_description
from bioimageio.core.prediction_pipeline import PredictionPipeline, create_prediction_pipeline

from tiktorch import log
from tiktorch.rpc import Shutdown
from tiktorch.rpc import mp as _mp_rpc
from tiktorch.rpc.mp import BioModelClient, MPServer

from ...converters import Sample
from .backend import base
from .rpc_interface import IRPCModelSession


class ModelSessionProcess(IRPCModelSession[PredictionPipeline]):
    def __init__(self, model: PredictionPipeline) -> None:
        super().__init__(model)
        self._datasets = {}
        self._worker = base.SessionBackend(self._model)

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
        input_specs=prediction_pipeline.input_specs, output_specs=prediction_pipeline.output_specs, api=api
    )


def _get_prediction_pipeline_from_model_bytes(model_zip: bytes, devices: List[str]) -> PredictionPipeline:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as _tmp_file:
        _tmp_file.write(model_zip)
        temp_file_path = pathlib.Path(_tmp_file.name)
    model = load_resource_description(temp_file_path)
    return create_prediction_pipeline(bioimageio_model=model, devices=devices)
