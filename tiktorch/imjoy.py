import asyncio
import base64
import logging
import zipfile
from pathlib import Path

import numpy
import torch
from imageio import imread

from tiktorch.server.reader import eval_model

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


class ImJoyPlugin:
    def setup(self) -> None:
        with zipfile.ZipFile("/g/kreshuk/beuttenm/Desktop/unet2d.model.zip", "r") as model_zip:  # todo: configure path
            self.exemplum = eval_model(
                model_file=model_zip, devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]
            )

        self.window = None
        api.log("initialized")

    async def run(self, ctx) -> None:
        image_path = Path("/g/kreshuk/beuttenm/data/cremi/sneak.png")  # todo: configure path
        try:
            await self.show_png(image_path)
        except Exception as e:
            logger.error(e)

        assert image_path.exists()
        img = imread(str(image_path))
        assert img.shape[2] == 4
        batch = img[None, :512, :512, 0]  # cyx

        prediction = self.exemplum.forward(batch)

        self.show_numpy(prediction)

    async def show_png(self, png_path: Path):
        with png_path.open("rb") as f:
            data = f.read()
            result = base64.b64encode(data).decode("ascii")

        imgurl = "data:image/png;base64," + result
        data = {"src": imgurl}

        data_plot = {"name": "show png", "type": "imjoy/image", "w": 12, "h": 15, "data": data}

        ## Check if window was defined
        if self.window is None:
            self.window = await api.createWindow(data_plot)
            print(f"Window created")

    def show_numpy(self, data: numpy.ndarray):
        print(data)


if __name__ == "__main__":
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
