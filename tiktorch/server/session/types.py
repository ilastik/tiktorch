from __future__ import annotations

import enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch


@enum.unique
class State(enum.Enum):
    Idle = "idle"
    Paused = "paused"
    Running = "running"
    Stopped = "stopped"


class Devices:
    def __init__(self):
        self.devices = []
        self.base_device = "cpu"

    def update(self, devices: List[torch.device]) -> List[torch.device]:
        free_devices = [d for d in self.devices if d not in devices]

        if not devices:
            self.base_device = "cpu"
            self.devices = []

        else:
            self.base_device = devices[0].type
            if not all(d.type == self.base_device for d in devices):
                raise ValueError("Can't train on cpu and gpu at the same time")

            self.devices = devices

        return free_devices

    def __len__(self):
        return len(self.devices)

    def __iter__(self):
        return iter(self.devices)
