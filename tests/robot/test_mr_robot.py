import numpy as np
from mr_robot.mr_robot import MrRobot
from tiktorch.server import TikTorchServer
from tiktorch.rpc import RPCFuture
from tiktorch.types import SetDeviceReturnType

import z5py

def test_MrRobot():
	robo = MrRobot()
	assert isinstance(robo,MrRobot)
	assert isinstance(robo.new_server, TikTorchServer)

	file = robo.load_data()
	assert isinstance(file, z5py.file.File)

	fut = robo.load_model()
	assert isinstance(fut, RPCFuture)

	robo.resume()
	op = robo.predict()

	assert op.shape == (1,1,32,32)
	assert isinstance(op, np.ndarray)
