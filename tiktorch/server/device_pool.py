from __future__ import annotations
import abc
import uuid

from typing import Dict, List
from collections import defaultdict

import threading
import enum
import torch


@enum.unique
class DeviceStatus(enum.Enum):
    AVAILABLE = "available"
    IN_USE = "in_use"


class IDevice(abc.ABC):
    @property
    @abc.abstractmethod
    def id(self) -> str:
        """
        Returns unique device id
        """
        ...

    @property
    @abc.abstractmethod
    def status(self) -> DeviceStatus:
        """
        Returns device status
        """
        ...


class ILease(abc.ABC):
    @property
    @abc.abstractmethod
    def id(self) -> str:
        """
        Returns unique lease id
        """
        ...

    @abc.abstractmethod
    def terminate(self) -> None:
        """
        Terminates lease
        """
        ...

    @property
    @abc.abstractmethod
    def devices(self) -> List[IDevice]:
        """
        Returns list of leased devices
        """
        ...


class IDevicePool(abc.ABC):
    @abc.abstractmethod
    def list_devices(self) -> List[IDevice]:
        """
        List devices available on server
        """
        ...

    @abc.abstractmethod
    def lease(self, device_ids: List[str]) -> ILease:
        """
        Lease devices for session
        """
        ...


class _Device(IDevice):
    def __init__(self, id_: str, status: DeviceStatus) -> None:
        self.__id = id_
        self.__status = status

    @property
    def id(self) -> str:
        return self.__id

    @property
    def status(self) -> DeviceStatus:
        return self.__status


class _Lease(ILease):
    def __init__(self, pool, id_: str) -> None:
        self.__id = id_
        self.__pool = pool

    @property
    def id(self) -> str:
        return self.__id

    @property
    def devices(self) -> List[IDevice]:
        return self.__pool._get_devices(self.__id)

    def terminate(self) -> None:
        self.__pool._release_devices(self.__id)


class TorchDevicePool(IDevicePool):
    def __init__(self):
        self.__lease_id_by_device_id = {}
        self.__device_ids_by_lease_id = defaultdict(list)
        self.__lock = threading.Lock()

    def list_devices(self) -> List[IDevice]:
        with self.__lock:
            ids = ["cpu"]

            if torch.cuda.is_available():
                ids += [f"gpu:{idx}" for idx in range(torch.cuda.device_count())]

            devices: List[IDevice] = []
            for id_ in ids:
                status = DeviceStatus.AVAILABLE
                if id_ in self.__lease_id_by_device_id:
                    status = DeviceStatus.IN_USE

                devices.append(_Device(id_=id_, status=status))

            return devices

    def lease(self, device_ids: List[str]) -> ILease:
        if not device_ids:
            raise Exception("No devices specified")

        with self.__lock:
            lease_id = uuid.uuid4().hex
            for dev_id in device_ids:
                if dev_id in self.__lease_id_by_device_id:
                    raise Exception(f"Device {dev_id} is already in use")

            for dev_id in device_ids:
                self.__lease_id_by_device_id[dev_id] = lease_id
                self.__device_ids_by_lease_id[lease_id].append(dev_id)

            return _Lease(self, id_=lease_id)

    def _get_lease_devices(self, lease_id: str) -> List[IDevice]:
        return [_Device(id_=dev_id, status=DeviceStatus.IN_USE) for dev_id in self.__device_ids_by_lease_id[lease_id]]

    def _release_devices(self, lease_id: str) -> None:
        with self.__lock:
            dev_ids = self.__device_ids_by_lease_id.pop(lease_id, [])

            for id_ in dev_ids:
                del self.__lease_id_by_device_id[id_]
