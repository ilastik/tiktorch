import multiprocessing as _mp

from typing import Tuple, Optional, List

from tiktorch.rpc import mp as _mp_rpc

from .interface import ISession
from .types import State

__all__ = ["start_session_process"]


def start_training_process(
    config: dict, model: "torch.nn.Module", optimizer_state: bytes = b"", log_queue: Optional[_mp.Queue] = None
) -> Tuple[_mp.Process, ISession]:
    from .process import run

    client_conn, server_conn = _mp.Pipe()
    proc = _mp.Process(
        target=run,
        name="Training",
        kwargs={
            "conn": server_conn,
            "config": config,
            "model": model,
            "optimizer_state": optimizer_state,
            "log_queue": log_queue,
        },
    )
    proc.start()
    return proc, _mp_rpc.create_client(ISession, client_conn)


def start_session_process(
    model_zip: bytes, devices: List[str], log_queue: Optional[_mp.Queue] = None
) -> Tuple[_mp.Process, ISession]:
    from .process import run_model_process

    client_conn, server_conn = _mp.Pipe()
    proc = _mp.Process(
        target=run_model_process,
        name="ModelProcess",
        kwargs={"conn": server_conn, "devices": devices, "log_queue": log_queue, "model_zip": model_zip},
    )
    proc.start()
    return proc, _mp_rpc.create_client(ISession, client_conn)
