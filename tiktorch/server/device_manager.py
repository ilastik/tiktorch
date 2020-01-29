from __future__ import annotations
import abc

from typing import List, Dict
from collections import defaultdict

import threading
import enum
import torch

from .session_manager import ISession


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


class IDeviceManager(abc.ABC):
    @abc.abstractmethod
    def list_devices(self) -> List[IDevice]:
        """
        List devices available on server
        """
        ...

    @abc.abstractmethod
    def lease(self, session, device_ids: List[str]) -> None:
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


class TorchDeviceManager(IDeviceManager):
    def __init__(self):
        self.__session_id_by_device_id = {}
        self.__device_ids_by_session_id = defaultdict(list)
        self.__lock = threading.Lock()

    def list_devices(self) -> List[IDevice]:
        with self.__lock:
            ids = ["cpu"]

            if torch.cuda.is_available():
                ids += [f"gpu:{idx}" for idx in range(torch.cuda.device_count())]

            devices = []
            for id_ in ids:
                status = DeviceStatus.AVAILABLE
                if id_ in self.__session_id_by_device_id:
                    status = DeviceStatus.IN_USE

                devices.append(_Device(id_=id_, status=status))

            return devices

    def list_session_devices(self, session: ISession) -> List[Device]:
        with self.__lock:
            dev_ids = self.__device_ids_by_session_id[session.id]
            return [_Device(id_=id_, status=DeviceStatus.IN_USE) for id_ in dev_ids]

    def lease(self, session: ISession, device_ids: List[str]) -> None:
        with self.__lock:
            for dev_id in device_ids:
                if dev_id in self.__session_id_by_device_id:
                    raise Exception(f"Device {dev_id} is already in use")

            for dev_id in device_ids:
                self.__session_id_by_device_id[dev_id] = session.id
                self.__device_ids_by_session_id[session.id].append(dev_id)

            session.on_close(self.__on_session_close)

    def __on_session_close(self, session: ISession) -> None:
        with self.__lock:
            dev_ids = self.__device_ids_by_session_id.pop(session.id, [])

            for id_ in dev_ids:
                del self.__session_id_by_device_id[id_]
