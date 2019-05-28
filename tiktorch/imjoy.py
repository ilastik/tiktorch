import logging
import os
from asyncio import Future

import numpy
import yaml

from typing import List, Optional, Tuple

from tiktorch.types import SetDeviceReturnType, NDArray
from tiktorch.server import TikTorchServer
from tiktorch.rpc import Shutdown, RPCFuture

logger = logging.getLogger(__name__)

try:
    from imjoy import api
except ImportError:
    class ImjoyApi:
        def log(self, msg) -> None:
            logger.info(msg)

        def alert(self, msg) -> None:
            logger.warning(msg)

        async def showDialog(self, *args, **kwargs) -> None:
            print(args, kwargs)

    api = ImjoyApi()




class ImJoyPlugin():
    def setup(self) -> None:
        self.server = TikTorchServer()
        api.log("initialized")

    async def run(self, ctx) -> None:
        # todo: remvoe  this (set through ui)
        ctx.config.config_folder = "/repos/tiktorch/tests/data/CREMI_DUNet_pretrained_new"
        available_devices = self.server.get_available_devices()
        api.log(f"available devices: {available_devices}")
        self.config = ctx.config
        await self._choose_devices(available_devices)

    async def _choose_devices(self, available_devices) -> None:
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
            "textOff": "Not Selected",
        }

        def fill_template(update: dict):
            ret = dict(device_switch_template)
            ret.update(update)
            return ret

        choose_devices_schema = {"fields": [fill_template({"model": d[0], "label": d[1]}) for d in available_devices]}
        self.dialog = await api.showDialog(
            {
                "name": "Select from available devices",
                "type": "SchemaIO",
                "w": 40,
                "h": 15,
                "data": {
                    "title": f"Select devices for TikTorch server",
                    "schema": choose_devices_schema,
                    "model": {},
                    "callback": self._choose_devices_callback,
                    "show": True,
                    "formOptions": {"validateAfterLoad": True, "validateAfterChanged": True},
                    "id": 0,
                },
            }
        )
        # self.dialog.onClose(self._choose_devices_close_callback)

    # def _choose_devices_close_callback(self) -> None:
    #     api.log("select device dialog closed")
    #     self._chosen_devices = []
    def _choose_devices_callback(self, data) -> None:
        api.log("before chosen devices callback")
        chosen_devices = [d for d, selected in data.items() if selected]
        api.log(f"chosen devices callback: {chosen_devices}")
        self.dialog.close()
        self._load_model(chosen_devices)

    def _load_model(self, chosen_devices) -> RPCFuture[SetDeviceReturnType]:
        # todo: select individual files through gui
        # load config
        config_file_name = os.path.join(self.config.config_folder, "tiktorch_config.yml")
        if not os.path.exists(config_file_name):
            raise FileNotFoundError(f"Config file not found at: {config_file_name}.")

        with open(config_file_name, "r") as f:
            tiktorch_config = yaml.load(f, Loader=yaml.SafeLoader)

        # Read model.py
        file_name = os.path.join(self.config.config_folder, "model.py")
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Model file not found at: {file_name}.")

        with open(file_name, "rb") as f:
            binary_model_file = f.read()

        # Read model and optimizer states if they exist
        binary_states = []
        for file_name in ["state.nn", "optimizer.nn"]:
            file_name = os.path.join(self.config.config_folder, file_name)
            if os.path.exists(file_name):
                with open(file_name, "rb") as f:
                    binary_states.append(f.read())
            else:
                binary_states.append(b"")

        return self.server.load_model(tiktorch_config, binary_model_file, *binary_states, devices=chosen_devices)

    async def forward(self, data: numpy.ndarray, id_: Optional[Tuple] = None) -> Tuple[numpy.ndarray, Optional[Tuple]]:
        tikfut = self.server.forward(NDArray(data, id_=id_))
        return tikfut.map(lambda x: (x.as_numpy(), id_)).result()

    async def exit(self):
        api.log("shutting down...")
        try:
            self.server.shutdown()
        except Shutdown:
            api.log("shutdown successful")
        else:
            api.log("shutdown failed")




if __name__ == "__main__":
    import asyncio
    from dataclasses import dataclass, field

    @dataclass
    class Config:
        pass

    @dataclass()
    class Ctx:
        config: Config = field(default_factory=Config)

    logging.basicConfig(level=logging.DEBUG)
    loop = asyncio.get_event_loop()
    ctx = Ctx()

    plugin = ImJoyPlugin()
    plugin.setup()
    loop.run_until_complete(plugin.run(ctx))
    loop.run_until_complete(plugin.exit())
