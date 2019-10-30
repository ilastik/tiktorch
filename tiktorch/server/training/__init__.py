import multiprocessing as _mp

from typing import Tuple, Optional

from tiktorch.rpc import mp as _mp_rpc

from .interface import ITraining

__all__ = ["run", "ITraining"]


def start_training_process(
    config: dict, model: "torch.nn.Module", optimizer_state: bytes = b"", log_queue: Optional[_mp.Queue] = None
) -> Tuple[_mp.Process, ITraining]:
    from .base import run

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
    return proc, _mp_rpc.create_client(ITraining, client_conn)
