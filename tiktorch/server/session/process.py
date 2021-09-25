import dataclasses
import multiprocessing as _mp
import os
import pathlib
import tempfile
import uuid
from concurrent.futures import Future
from multiprocessing.connection import Connection
from typing import List, Optional, Tuple, Union

import numpy
from bioimageio.core import load_resource_description
from bioimageio.core.prediction_pipeline import PredictionPipeline, create_prediction_pipeline
from bioimageio.spec.shared.raw_nodes import ImplicitOutputShape, ParametrizedInputShape

from tiktorch import log
from tiktorch.rpc import Shutdown
from tiktorch.rpc import mp as _mp_rpc
from tiktorch.rpc.mp import MPServer

from .backend import base
from .rpc_interface import IRPCModelSession

# pairs of axis-shape for a single tensor
NamedInt = Tuple[str, int]
NamedFloat = Tuple[str, float]
NamedShape = List[NamedInt]
NamedVec = List[NamedFloat]


@dataclasses.dataclass
class NamedParametrizedShape:
    min_shape: NamedShape
    step_shape: NamedShape


@dataclasses.dataclass
class ModelInfo:
    """Intermediate representation of bioimageio neural network model

    TODO (k-dominik): ModelInfo only used in inference_servicer to convert to
    protobuf modelinfo.

    """

    # TODO: Test for model info
    name: str
    input_axes: List[str]  # one per input
    output_axes: List[str]  # one per output
    input_shapes: List[Union[NamedShape, NamedParametrizedShape]]  # per input multiple shapes
    halos: List[NamedShape]  # one per output
    offsets: List[NamedShape]  # one per output
    scales: List[NamedVec]  # one per output

    @classmethod
    def from_prediction_pipeline(cls, prediction_pipeline: PredictionPipeline):
        if not all(
            isinstance(output_spec.shape, ImplicitOutputShape) for output_spec in prediction_pipeline.output_specs
        ):
            raise NotImplementedError("Currently only implicit output shapes are supported v0v")

        input_shapes = []
        for input_spec in prediction_pipeline.input_specs:
            if isinstance(input_spec.shape, ParametrizedInputShape):
                input_shapes.append(
                    NamedParametrizedShape(
                        min_shape=list(map(tuple, zip(input_spec.axes, input_spec.shape.min))),
                        step_shape=list(map(tuple, zip(input_spec.axes, input_spec.shape.step))),
                    )
                )
            else:
                input_shapes.append(list(map(tuple, zip(input_spec.axes, input_spec.shape))))

        halos = [
            list(map(tuple, zip(output_spec.axes, output_spec.halo)))
            for output_spec in prediction_pipeline.output_specs
        ]

        scales = [
            list(map(tuple, zip(output_spec.axes, output_spec.shape.scale)))
            for output_spec in prediction_pipeline.output_specs
        ]
        offsets = [
            list(map(tuple, zip(output_spec.axes, output_spec.shape.offset)))
            for output_spec in prediction_pipeline.output_specs
        ]
        return cls(
            name=prediction_pipeline.name,
            input_axes=["".join(input_spec.axes) for input_spec in prediction_pipeline.input_specs],
            output_axes=["".join(output_spec.axes) for output_spec in prediction_pipeline.output_specs],
            input_shapes=input_shapes,
            halos=halos,
            scales=scales,
            offsets=offsets,
        )


class ModelSessionProcess(IRPCModelSession):
    def __init__(self, model_zip: bytes, devices: List[str]) -> None:
        _tmp_file = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        _tmp_file.write(model_zip)
        _tmp_file.close()
        model = load_resource_description(pathlib.Path(_tmp_file.name))
        os.unlink(_tmp_file.name)
        self._model: PredictionPipeline = create_prediction_pipeline(bioimageio_model=model, devices=devices)
        self._datasets = {}
        self._worker = base.SessionBackend(self._model)

    def forward(self, input_tensors: numpy.ndarray) -> Future:
        res = self._worker.forward(input_tensors)
        return res

    def create_dataset(self, mean, stddev):
        id_ = uuid.uuid4().hex
        self._datasets[id_] = {"mean": mean, "stddev": stddev}
        return id_

    def get_model_info(self) -> ModelInfo:
        return ModelInfo.from_prediction_pipeline(self._model)

    def shutdown(self) -> Shutdown:
        self._worker.shutdown()
        return Shutdown()


def _run_model_session_process(
    conn: Connection, model_zip: bytes, devices: List[str], log_queue: Optional[_mp.Queue] = None
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

    session_proc = ModelSessionProcess(model_zip, devices)
    srv = MPServer(session_proc, conn)
    srv.listen()


def start_model_session_process(
    model_zip: bytes, devices: List[str], log_queue: Optional[_mp.Queue] = None
) -> Tuple[_mp.Process, IRPCModelSession]:
    client_conn, server_conn = _mp.Pipe()
    proc = _mp.Process(
        target=_run_model_session_process,
        name="ModelSessionProcess",
        kwargs={"conn": server_conn, "devices": devices, "log_queue": log_queue, "model_zip": model_zip},
    )
    proc.start()
    return proc, _mp_rpc.create_client(IRPCModelSession, client_conn)
