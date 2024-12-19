from tiktorch.proto import utils_pb2
from tiktorch.server.device_pool import DeviceStatus, IDevicePool


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
