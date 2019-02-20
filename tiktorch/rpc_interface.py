from tiktorch.rpc import RPCInterface, exposed
from tiktorch.types import NDArrayBatch


class IFlightControl(RPCInterface):
    @exposed
    def ping(self) -> bytes:
        raise NotImplementedError

    @exposed
    def shutdown(self) -> None:
        raise NotImplementedError


class INeuralNetworkAPI(RPCInterface):
    @exposed
    def load_model(
        self,
        config: dict,
        model_file: bytes,
        model_state: bytes,
        optimizer_state: bytes,
    ) -> None:
        raise NotImplementedError

    @exposed
    def set_hparams(self, params: dict) -> None:
        raise NotImplementedError

    @exposed
    def forward(self, batch: NDArrayBatch) -> NDArrayBatch:
        raise NotImplementedError

    @exposed
    def pause(self) -> None:
        raise NotImplementedError

    @exposed
    def resume(self) -> None:
        raise NotImplementedError

    @exposed
    def training_process_is_running(self) -> bool:
        raise NotImplementedError

    @exposed
    def dry_run(self, conf: dict) -> dict:
        raise NotImplementedError

    @exposed
    def train(self, data: NDArrayBatch, labels: NDArrayBatch) -> None:
        raise NotImplementedError
