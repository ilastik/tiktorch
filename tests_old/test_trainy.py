import logging
import time
import unittest

import numpy as np
import torch.nn as nn

from tiktorch.wrapper import TikTorch

logging.basicConfig(level=logging.INFO)


class TestTrainy(unittest.TestCase):
    MODEL = "LITE"

    def setUp(self):
        TikTorch.read_config = lambda self: self
        tiktorch = TikTorch(build_directory=".")
        if self.MODEL == "HEAVY":
            tiktorch._model = nn.Sequential(
                nn.Conv2d(1, 512, 9, padding=4),
                nn.ELU(),
                nn.Conv2d(512, 512, 9, padding=4),
                nn.ELU(),
                nn.Conv2d(512, 512, 9, padding=4),
                nn.ELU(),
                nn.Conv2d(512, 512, 9, padding=4),
                nn.ELU(),
                nn.Conv2d(512, 1, 9, padding=4),
            )
        elif self.MODEL == "LITE":
            tiktorch._model = nn.Conv2d(1, 1, 1)
        tiktorch._config = {"input_shape": [1, 512, 512], "dynamic_input_shape": "(32 * (nH + 1), 32 * (nW + 1))"}
        tiktorch._set_handler(tiktorch._model)
        self.tiktorch = tiktorch

    def test_forward(self):
        logger = logging.getLogger("test_forward")
        out = self.tiktorch.forward([np.random.uniform(size=(512, 512)).astype("float32") for _ in range(3)])
        logger.info(f"out.shape = {out.shape}")

    def test_train(self):
        logger = logging.getLogger("test_train")
        # Start session
        mock_data = [np.random.uniform(size=(1, 320, 320)).astype("float32") for _ in range(3)]
        mock_label = [np.random.randint(0, 2, size=(1, 320, 320)).astype("float32") for _ in range(3)]
        logger.info(f"Starting Training")
        self.tiktorch.handler.train(mock_data, mock_label)
        # Infer in parallel
        time.sleep(5)
        for it in range(10):
            logger.info(f"Inference Iter: {it}")
            self.tiktorch.forward([np.random.uniform(size=(512, 512)).astype("float32") for _ in range(3)])
            time.sleep(2)
        logger.info(f"Sending Stop Signal")
        self.tiktorch.handler.stop_training()


if __name__ == "__main__":
    unittest.main()
