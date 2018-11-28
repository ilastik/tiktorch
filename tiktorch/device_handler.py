from itertools import count
import logging

import torch
import numpy as np

from tiktorch.utils import DynamicShape, assert_, to_list
from tiktorch.blockinator import Blockinator, th_pad
from tiktorch.trainy import Trainer
# from .dataloader import get_dataloader
# from .trainer import TikTorchTrainer

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
    def __init__(self, *, model, device_names, channels, dynamic_shape_code,
                 training_hyperparams=None):
        # Privates
        self._model = None
        self._trainer = None
        self._max_batch_limit = 500
        self._halo = None
        self._channels = channels
        self._device_specs = {}
        self.__num_trial_runs_on_device = {}
        # Publics
        self.device_names = to_list(device_names)
        self.dynamic_shape = DynamicShape(dynamic_shape_code)
        # Set
        self._set_model(model)
        self._set_trainer(training_hyperparams)
        # Init superclass
        super(ModelHandler, self).__init__(num_parallel_jobs=len(self.devices))

    @property
    def channels(self):
        return self._channels

    @property
    def model(self):
        assert self._model is not None
        return self._model

    def _set_model(self, model):
        # Use this only once to prevent amusing bugs
        assert self._model is None
        self._model = model.to(self.device)

    def _set_trainer(self, hyperparameters):
        self._trainer = Trainer(handler=self,
                                hyperparameters=hyperparameters)

    @property
    def trainer(self):
        assert self._trainer is not None
        return self._trainer

    @property
    def device(self):
        return torch.device(self.device_names[0])

    @property
    def devices(self):
        return [torch.device(name) for name in self.device_names]

    @property
    def num_devices(self):
        return len(self.device_names)

    def get_device_spec(self, device_id):
        device_spec = self._device_specs.get(device_id)
        assert_(device_spec is not None,
                f"device_id {device_id} not found in specs. Consider calling dry_run() first.",
                RuntimeError)
        return device_spec

    @property
    def halo_in_blocks(self):
        """
        Returns a list, containing the number of dynamic base shape blocks to cover the halo.
        """
        return [int(np.ceil(_halo / _block_shape))
                for _halo, _block_shape in zip(self.halo, self.dynamic_shape.base_shape)]
    
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

    def train(self, data, labels):
        self.trainer.push(data, labels)
        return self

    def pause_training(self):
        self.trainer.pause()
        return self

    def resume_training(self):
        self.trainer.resume()
        return self

    def stop_training(self):
        self.trainer.shut_down_training_process()
        return self

    def start_training(self):
        self.trainer.ignition()
        return self

    def dump_state(self, filename):
        state_dict = self.model.state_dict()
        torch.save(state_dict, filename)
        return self

    def _train_trial_run_successful(self, *input_shape, device_id=None):
        if device_id is None:
            return [self._trial_run_successful(*input_shape, device_id=_device_id)
                    for _device_id in range(len(self.devices))]
        try:
            if device_id not in self.__num_trial_runs_on_device:
                self.__num_trial_runs_on_device[device_id] = 1
            else:
                self.__num_trial_runs_on_device[device_id] += 1
            device = self.devices[device_id]
            self.model.to(device)(torch.zeros(1, *input_shape).to(device))
            return True
        except RuntimeError:
            # FIXME Investigate
            # Unexplained torch weirdness: new device can be "out of memory" for no reason,
            # so we give it another chance
            if self.__num_trial_runs_on_device[device_id] == 1:
                # second chance
                return self._trial_run_successful(*input_shape, device_id=device_id)
            else:
                # Nope
                return False

    def _trial_run_successful(self, *input_shape, device_id=None):
        if device_id is None:
            return [self._trial_run_successful(*input_shape, device_id=_device_id)
                    for _device_id in range(len(self.devices))]
        try:
            if device_id not in self.__num_trial_runs_on_device:
                self.__num_trial_runs_on_device[device_id] = 1
            else:
                self.__num_trial_runs_on_device[device_id] += 1
            with torch.no_grad():
                device = self.devices[device_id]
                self.model.to(device)(torch.zeros(1, *input_shape).to(device))
            return True
        except RuntimeError:
            # FIXME Investigate
            # Unexplained torch weirdness: new device can be "out of memory" for no reason,
            # so we give it another chance
            if self.__num_trial_runs_on_device[device_id] == 1:
                # second chance
                return self._trial_run_successful(*input_shape, device_id=device_id)
            else:
                # Nope
                return False

    def _try_running_on_blocksize(self, *block_size, device_id, train_flag=False):
        if train_flag:
            return self._train_trial_run_successful(self.channels, *self.dynamic_shape(*block_size),
                                                    device_id=device_id)
        else:
            return self._trial_run_successful(self.channels, *self.dynamic_shape(*block_size),
                                              device_id=device_id)

    def _dry_run_on_device(self, device_id=0):
        # Sweep diagonals to find where it crashes.
        max_diagonal_count = 0
        previous_spatial_shape = []
        for diagonal_count in count():
            # Check if the spatial shape has not changed. If it hasn't, it means the user doesn't
            # want dynamic shapes.
            spatial_shape = self.dynamic_shape(*([diagonal_count] * len(self.dynamic_shape)))
            logger.debug(f"Diagonal sweep (GPU{device_id}) iteration {diagonal_count}; "
                         f"shape = {spatial_shape}.")
            if previous_spatial_shape == spatial_shape:
                break
            else:
                previous_spatial_shape = spatial_shape
            success = self._try_running_on_blocksize(*([diagonal_count] * len(self.dynamic_shape)),
                                                     device_id=device_id)
            if not success:
                # GPU borked, no more
                logger.debug(f"GPU{device_id} borked at diagonal iteration {diagonal_count}; "
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
                logger.debug(f'Non-diagonal sweep (GPU{device_id}): '
                             f'dimension: {dim_num}; block num: {block_count}')
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
                    logger.debug(f'GPU{device_id} borked at non-diagonal sweep: dimension: '
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

    def dry_run(self):
        for device_id in range(self.num_devices):
            logger.debug(f'Dry running on device: {device_id}')
            self._device_specs[device_id] = self._dry_run_on_device(device_id)
        return self

    def binary_dry_run(self, image_shape, train_flag=False):
        """
        Parameters
        ----------
        image_shape: list or tuple
        """
        image_shape = list(image_shape) # in case image_shape is a tuple
        assert len(image_shape) == len(self.dynamic_shape.base_shape)
        
        if self.devices[0] == torch.device('cpu'):
            default_shape = [256, 256] if len(image_shape) == 2 else [96, 96, 96]
            image_shape = [default_shape[i] if image_shape[i] > default_shape[i] else image_shape[i]
                           for i in range(len(image_shape))]

        logger.debug(f'Dry run with upper bound: {image_shape}')
            
        max_shape = []
        for device_id in range(self.num_devices):
            logger.debug(f'Dry running on device: {self.devices[device_id]}')
            self._device_specs[device_id] = self._binary_dry_run_on_device(image_shape, device_id, train_flag=train_flag)
            max_device_shape = self.dynamic_shape(*self._device_specs[device_id].num_blocks)
            if len(max_shape) == 0:
                max_shape = max_device_shape
            elif max_shape < max_device_shape:
                max_shape = max_device_shape
        logger.debug(f'Dry run finished. Max shape / upper bound: {max_shape} / {image_shape}')
        return max_shape

    def _binary_dry_run_on_device(self, image_shape, device_id, train_flag=False):
        """
        Parameters
        ----------
        max_shape: list in base shape units
        """
        ndim_image = len(image_shape)
        previous_spatial_shape = [0 for i in range(ndim_image)]
        device_capacity = [0 for i in range(ndim_image)]
        l = [0 for _ in range(ndim_image)]
        r = [int(np.ceil(size / base_shape))
             for size, base_shape in zip(image_shape, self.dynamic_shape.base_shape)]
        m = [int(np.floor((l[i] + r[i]) / 2)) for i in range(ndim_image)]
        bark = False
        break_flag = False

        while sum(l) <= sum(r):
            for i in range(ndim_image):
                m = [int(np.floor((l[i] + r[i]) / 2)) for i in range(ndim_image)]
                spatial_shape = self.dynamic_shape(*m)
                
                logger.debug(f"Dry run on ({self.devices[device_id]}) with shape = {spatial_shape}.")
                print(f"Dry run on ({self.devices[device_id]}) with shape = {spatial_shape}.")

                if spatial_shape > image_shape:
                    break_flag = True
                    break
                else:
                    device_capacity = m
                
                success = self._try_running_on_blocksize(*m, device_id=device_id, train_flag=train_flag)

                if not success:
                    logger.debug(f"{self.devices[device_id]} barked at shape = {spatial_shape}.")
                    bark = True
                    r[i] = m[i] - 1 if m[i] - 1 > 0 else m[i]
                elif success and bark is False:
                    l[i] = m[i] + 1
                else:
                    device_capacity = m
                    break_flag = True
                    break

            if previous_spatial_shape == spatial_shape:
                break
            else:
                previous_spatial_shape = spatial_shape

            if break_flag == True:
                break

        return DeviceMemoryCapacity(device_capacity, self.dynamic_shape, device_id=device_id)
           
    @property
    def num_parallel_jobs(self):
        return self.num_devices

    def compute_halo(self, device_id=0, set_=True):
        device = self.devices[device_id]
        # Evaluate model on the smallest possible image to keep it quick
        input_tensor = torch.zeros(1, self.channels, *self.dynamic_shape.base_shape)
        output_tensor = self.model.to(device)(input_tensor.to(device))
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
        if set_:
            self.halo = halo
        return halo

    def crop_output_tensor(self, tensor, num_channel_axes=2):
        base_shape = self.dynamic_shape.base_shape
        roi_shape = []
        for size, halo, blocks in zip(base_shape, self.halo, self.halo_in_blocks):
            if halo > 0:
                roi_shape.append(slice(size * blocks, -size * blocks))
            else:
                roi_shape.append(slice(None))
        return tensor[[slice(None)] * num_channel_axes + roi_shape]

    def crop_halo(self, tensor, num_channel_axes=2):
        base_shape = self.dynamic_shape.base_shape
        roi_shape = []
        for size, halo, blocks in zip(base_shape, self.halo, self.halo_in_blocks):
            if halo > 0:
                roi_shape.append(slice(size*blocks - halo, -(size*blocks - halo)))
            else:
                roi_shape.append(slice(None))
        return tensor[[slice(None)] * num_channel_axes + roi_shape]

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: torch.Tensor
        """
        try:
            self.model.to(self.device)(torch.zeros(1, self.channels, *self.dynamic_shape.base_shape).to(self.device))
        except:
            logger.debug(f"Can't load tensor on `{self.device}`")
            RuntimeError(f"Can't load tensor on `{self.device}`")
        block = Blockinator(input_tensor, self.dynamic_shape.base_shape,
                            num_channel_axes=2, pad_fn=th_pad)
        with block.attach(self):
            output_tensor = block.process()
        return output_tensor

    def to_device(self, obj):
        return obj.to(self.device)

    def process_tensor(self, tensor):
        tensor = self.to_device(tensor)
        return self.model(tensor)