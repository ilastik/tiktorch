from typing import TypeVar

import grpc

from tiktorch.proto import utils_pb2
from tiktorch.server.device_pool import DeviceStatus, IDevicePool
from tiktorch.server.session_manager import Session, SessionManager


def list_devices(device_pool: IDevicePool) -> utils_pb2.Devices:
    devices = device_pool.list_devices()
    pb_devices = []
    for dev in devices:
        if dev.status == DeviceStatus.AVAILABLE:
            pb_status = utils_pb2.Device.Status.AVAILABLE
        elif dev.status == DeviceStatus.IN_USE:
            pb_status = utils_pb2.Device.Status.IN_USE
        else:
            raise ValueError(f"Unknown status value {dev.status}")

        pb_devices.append(utils_pb2.Device(id=dev.id, status=pb_status))

    return utils_pb2.Devices(devices=pb_devices)


T = TypeVar("T")


def get_model_session(
    session_manager: SessionManager[T], model_session_id: utils_pb2.ModelSession, context
) -> Session[T]:
    session = session_manager.get(model_session_id.id)

    if session is None:
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"model-session with id {model_session_id.id} doesn't exist")

    return session
