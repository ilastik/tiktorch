import logging
import socket

from imjoy import api

from tiktorch.launcher import LocalServerLauncher, RemoteSSHServerLauncher, SSHCred
from tiktorch.rpc_interface import INeuralNetworkAPI
from tiktorch.rpc import Client, TCPConnConf

logger = logging.getLogger(__name__)


class ImJoyPlugin(INeuralNetworkAPI):
    def setup(self):
        api.log('initialized')

    async def _choose_devices(self, data):
        await api.alert(str(data))

    async def run(self, ctx):
        address = socket.gethostbyname(ctx.config.address)
        port1 = str(ctx.config.port1)
        port2 = str(ctx.config.port2)

        conn_conf = TCPConnConf(address, port1, port2)

        if address == "127.0.0.1":
            self.launcher = LocalServerLauncher(conn_conf)
        else:
            self.launcher = RemoteSSHServerLauncher(
                conn_conf, cred=SSHCred(ctx.config.username, key_path=ctx.config.key_path)
            )

        api.log(f"start server at {address}:{port1};{port2}")
        self.launcher.start()
        api.alert("server running at {address}:{port1};{port2}")
        try:
            tikTorchClient = Client(INeuralNetworkAPI(), conn_conf)
            available_devices = tikTorchClient.get_available_devices()
        except Exception as e:
            self.launcher.stop()
            logger.exception(e)
            return

        api.log(f"available devices: {available_devices}")
        device_switch_template = {
                "type": "switch",
                "label": "Device",
                "model": "status",
                "multi": True,
                "readonly": False,
                "featured": False,
                "disabled": False,
                "default": False,
                "textOn": "Selected",
                "textOff": "Not Selected"
            }

        choose_devices_schema = {
           "fields": [
            device_switch_template,
            ]
        }
        answer = await api.showDialog({
           "name": 'Select from available devices',
           "type": 'SchemaIO',
           "w": 40,
           "h": 15,
           "data": {
               "title": f"Select devices for TikTorch server at {address}",
               "schema": choose_devices_schema,
               "model": {},
               "callback": self._choose_devices,
               "show": True,
               "formOptions": {
                   "validateAfterLoad": True,
                   "validateAfterChanged": True},
               "id": 0}})


    def exit(self):
        self.launcher.stop()
        api.log("tiktorch server stopped")
