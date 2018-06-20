import torch
from tiktorch.utils import DynamicShape, assert_, to_list
from itertools import count
import logging

logger = logging.getLogger('DeviceHandler')


class DeviceMemoryCapacity(object):
    def __init__(self, num_blocks, dynamic_shape, device_id=None):
        self.num_blocks = num_blocks
        self.dynamic_shape = dynamic_shape
        self.device_id = device_id

    @property
    def shape(self):
        return self.dynamic_shape(*self.num_blocks)

    def is_device_capable(self, shape):
        # TODO
        pass

    def __repr__(self):
        return f"DeviceMemoryCapacity(max_shape={self.shape}, num_blocks={self.num_blocks})"


class Processor(object):
    def __init__(self, num_parallel_jobs=0):
        self._num_parallel_jobs = num_parallel_jobs

    @property
    def num_parallel_jobs(self):
        return self._num_parallel_jobs


class ModelHandler(Processor):
    def __init__(self, *, model, device_names, in_channels, out_channels=1,
                 dynamic_shape_code):
        # Privates
        self._max_batch_limit = 500
        self._halo = None
        self._model = model
        self._in_channels = in_channels
        self._out_channels = out_channels
        # Publics
        self.device_names = to_list(device_names)
        self.dynamic_shape = DynamicShape(dynamic_shape_code)
        # Init superclass
        super(ModelHandler, self).__init__(num_parallel_jobs=len(self.devices))

    @property
    def model(self):
        return self._model

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def device(self):
        return torch.device(self.device_names[0])

    @property
    def devices(self):
        return [torch.device(name) for name in self.device_names]

    @property
    def halo(self):
        if self._halo is None:
            self._halo = self.compute_halo()
        return self._halo

    @halo.setter
    def halo(self, value):
        if isinstance(value, int):
            self._halo = [value] * len(self.dynamic_shape)
        else:
            assert_(len(value) == len(self.dynamic_shape),
                    f"Halo of a {len(self.dynamic_shape)}-D network cannot "
                    f"be {len(value)}-D.",
                    ValueError)
            self._halo = value

    def _trial_run_successful(self, *input_shape, device_id=None):
        if device_id is None:
            return [self._trial_run_successful(*input_shape, device_id=_device_id)
                    for _device_id in range(len(self.devices))]
        try:
            with torch.no_grad():
                device = self.devices[device_id]
                self.model.to(device)(torch.zeros(1, *input_shape).to(device))
            return True
        except RuntimeError:
            # Nope
            return False

    def _try_running_on_blocksize(self, *block_size, device_id):
        return self._trial_run_successful(self.in_channels, *self.dynamic_shape(*block_size),
                                          device_id=device_id)

    def dry_run_on_device(self, device_id=0):
        # Sweep diagonals to find where it crashes.
        max_diagonal_count = 0
        previous_spatial_shape = []
        for diagonal_count in count():
            # Check if the spatial shape has not changed. If it hasn't, it means the user doesn't
            # want dynamic shapes.
            spatial_shape = self.dynamic_shape(*([diagonal_count] * len(self.dynamic_shape)))
            logger.debug(f"Diagonal sweep iteration {diagonal_count}; "
                         f"shape = {spatial_shape}.")
            if previous_spatial_shape == spatial_shape:
                break
            else:
                previous_spatial_shape = spatial_shape
            # FIXME Try for all devices
            success = self._try_running_on_blocksize(*([diagonal_count] * len(self.dynamic_shape)),
                                                     device_id=device_id)
            if not success:
                # GPU borked, no more
                logger.debug(f"GPU borked at diagonal iteration {diagonal_count}; "
                             f"shape = {spatial_shape}.")
                break
            else:
                # moar!
                max_diagonal_count = diagonal_count
        # So the GPU whines at say (n + 1, n + 1, n + 1) for n = max_diagonal_count.
        # We try to figure out which dimension is responsible by:
        # (a) keeping (n, n, n) as the diagonals, but
        # (b) for each dimension increment the number of blocks till it breaks.
        # We're looking for the number of extra blocks (in addition to the diagonals) required
        # to bork the GPU.
        num_extra_blocks = [0] * len(self.dynamic_shape)
        for dim_num in range(len(self.dynamic_shape)):
            previous_spatial_shape = []
            for block_count in count():
                if block_count == 0:
                    continue
                logger.debug(f'Non-diagonal sweep: dimension: {dim_num}; block num: {block_count}')
                num_blocks = [max_diagonal_count] * len(self.dynamic_shape)
                num_blocks[dim_num] += block_count
                spatial_shape = self.dynamic_shape(*num_blocks)
                # Check if the shape has actually changed
                if previous_spatial_shape == spatial_shape:
                    # Nope - nothing to do then.
                    break
                else:
                    # Yep - update previous spatial shape just to be sure
                    previous_spatial_shape = spatial_shape
                success = self._try_running_on_blocksize(*num_blocks, device_id=device_id)
                if success:
                    # okidokie, we can do more
                    num_extra_blocks[dim_num] = block_count
                else:
                    # GPU is borked, no more
                    logger.debug(f'GPU borked at non-diagonal sweep: dimension: '
                                 f'{dim_num}; block num: {block_count}')
                    break
        # Get total size supported by the GPU. For this, we only retain the extras
        device_capacity = [max_diagonal_count] * len(self.dynamic_shape)
        for _extra_block_dim, _extra_blocks in enumerate(num_extra_blocks):
            if _extra_blocks == max(num_extra_blocks):
                device_capacity[_extra_block_dim] += _extra_blocks
                break
            else:
                continue
        # Done
        return DeviceMemoryCapacity(device_capacity, self.dynamic_shape, device_id=device_id)

    def compute_halo(self, device_id=0):
        device = self.devices[device_id]
        # Evaluate model on the smallest possible image to keep it quick
        input_tensor = torch.zeros(1, self.in_channels, *self.dynamic_shape.base_shape).to(device)
        model = self.model.to(device)
        output_tensor = model(input_tensor)
        # Assuming NCHW or NCDHW, the first two axes are not relevant for computing halo
        input_spatial_shape = input_tensor.shape[2:]
        output_spatial_shape = output_tensor.shape[2:]
        shape_difference = [_ishape - _oshape
                            for _ishape, _oshape in zip(input_spatial_shape, output_spatial_shape)]
        # Support for only symmetric halos for now
        assert_(all(_shape_diff % 2 == 0 for _shape_diff in shape_difference),
                "Only symmetric halos are supported.", RuntimeError)
        # Compute halo
        halo = [_shape_diff // 2 for _shape_diff in shape_difference]
        return halo

    def forward(self, input_tensor):

        pass


def test_dry_run():
    import torch.nn as nn
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    model = nn.Sequential(nn.Conv2d(3, 512, 3),
                          nn.Conv2d(512, 512, 3),
                          nn.Conv2d(512, 512, 3),
                          nn.Conv2d(512, 3, 3))
    handler = ModelHandler(model=model,
                           device_names='cuda:0',
                           in_channels=3, out_channels=3,
                           dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
    spec = handler.dry_run_on_device(0)
    print(f"GPU Specs: {spec}")


def test_halo_computer():
    import torch.nn as nn
    model = nn.Sequential(nn.Conv2d(3, 10, 3),
                          nn.Conv2d(10, 10, 3),
                          nn.Conv2d(10, 10, 3),
                          nn.Conv2d(10, 3, 3))
    handler = ModelHandler(model=model,
                           device_names='cuda:0',
                           in_channels=3, out_channels=3,
                           dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
    print(f"Halo: {handler.halo}")


if __name__ == '__main__':
    # test_dry_run()
    test_halo_computer()
