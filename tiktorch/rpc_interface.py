from tiktorch.rpc import RPCInterface
from tiktorch.types import NDArrayBatch


class INeuralNetworkAPI(RPCInterface):
    def load_model(
        self,
        config: dict,
        model_file: bytes,
        model_state: bytes,
        optimizer_state: bytes,
    ) -> None:
        raise NotImplementedError

    def set_hparams(self, params: dict) -> None:
        raise NotImplementedError

    def ping(self) -> bytes:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError

    def forward(self, batch: NDArrayBatch) -> NDArrayBatch:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def resume(self) -> None:
        raise NotImplementedError

    def training_process_is_running(self) -> bool:
        raise NotImplementedError

    def train(self, data: NDArrayBatch, labels: NDArrayBatch) -> None:
        raise NotImplementedError
