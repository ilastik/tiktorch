from typing import List

import torch


class IDeviceManager:
    def list_available_devices(self):
        raise NotImplementedError()

    def reserve_devices(self, session, device_ids: List[str]) -> None:
        raise NotImplementedError()


class TorchDeviceManager(IDeviceManager):
    def __init__(self):
        pass

    def list_available_devices(self, session=None):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += [f"gpu:{idx}" for idx in range(torch.cuda.device_count())]
        return devices

    def reserve_devices(self, session, devices):
        pass
