import pytest
import torch.cuda
import torch.version

from tiktorch.server.device_pool import TorchDevicePool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_device_pool_with_cuda():
    device_pool = TorchDevicePool()
    assert device_pool.cuda_version == torch.version.cuda


@pytest.mark.skipif(torch.cuda.is_available(), reason="cuda is available")
def test_device_pool_without_cuda():
    device_pool = TorchDevicePool()
    assert device_pool.cuda_version is None
