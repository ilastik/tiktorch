import logging

from inferno.trainers import Trainer as InfernoTrainer

from torch.utils.data import DataLoader

from tiktorch.server.datasets import DynamicDataLoaderWrapper, DynamicDataset

logger = logging.getLogger(__name__)


class TikTrainer(InfernoTrainer):
    _ALIASES = {"training": "train", "validation": "validate"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._break_cb = None
        self._dataset_by_name = {}

    def set_break_callback(self, callback):
        self._break_cb = callback

    def get_dataset(self, name: str) -> DynamicDataset:
        return self._dataset_by_name[name]

    @property
    def max_num_iterations(self):
        return self._max_num_iterations

    def stop_fitting(self, max_num_iterations=None, max_num_epochs=None):
        if self._break_cb and self._break_cb():
            return True
        else:
            return super().stop_fitting(max_num_iterations=max_num_iterations, max_num_epochs=max_num_epochs)

    @classmethod
    def build(cls, *args, dataset_by_name, **kwargs):
        trainer = super().build(*args, **kwargs)

        trainer._dataset_by_name = dataset_by_name

        for name, dataset in dataset_by_name.items():
            name = cls._ALIASES.get(name, name)
            loader = DataLoader(dataset=dataset)
            trainer.bind_loader(name, DynamicDataLoaderWrapper(loader))

        return trainer

    def move_to(self, devices):
        if devices.base_device == "cpu":
            self.cpu()
        elif devices.base_device == "cuda":
            self.cuda(devices=[d.index for d in devices])
        else:
            raise ValueError(f"Unknown device type {devices.base_device}")

        # make sure optimizer states are on correct device
        for k in self.optimizer.state.keys():
            param_state = self.optimizer.state[k]
            for p in param_state.keys():
                try:
                    if not isinstance(param_state[p], int):
                        param_state[p] = param_state[p].to(devices.base_device)
                except Exception as e:
                    self.logger.exception("Failed to move optimizer to %s", devices)
