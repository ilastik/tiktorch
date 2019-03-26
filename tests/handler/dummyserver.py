import logging
import torch

from torch.multiprocessing import Pipe
from typing import Union, Tuple

from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl
from tiktorch.handler import HandlerProcess
from tiktorch.handler.constants import SHUTDOWN, SHUTDOWN_ANSWER
from tiktorch.types import NDArray, NDArrayBatch
from tiktorch.tiktypes import TikTensor, TikTensorBatch

logger = logging.getLogger(__name__)



class DummyServer(INeuralNetworkAPI, IFlightControl):
    def __init__(self, **kwargs):
        self.handler_conn, server_conn = Pipe()
        self.handler = HandlerProcess(server_conn=server_conn, **kwargs)
        self.handler.start()

    def forward(self, batch: TikTensor) -> None:
        pass

        # self.handler_conn.send(
        #     (
        #         "forward",
        #         {"keys": [a.id for a in batch], "data": torch.stack([torch.from_numpy(a.as_numpy()) for a in batch])},
        #     )
        # )

    def active_children(self):
        self.handler_conn.send(("active_children", {}))

    def listen(self, timeout: float = 10) -> Union[None, Tuple[str, dict]]:
        if self.handler_conn.poll(timeout=timeout):
            answer = self.handler_conn.recv()
            logger.debug("got answer: %s", answer)
            return answer
        else:
            return None

    def shutdown(self):
        self.handler_conn.send(SHUTDOWN)
        got_shutdown_answer = False
        while self.handler.is_alive():
            if self.handler_conn.poll(timeout=2):
                answer = self.handler_conn.recv()
                if answer == SHUTDOWN_ANSWER:
                    got_shutdown_answer = True

        assert got_shutdown_answer
