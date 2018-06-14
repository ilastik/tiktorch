import torch
from .utils import DynamicShape, assert_, to_list


class BatchSpec(object):
    def __init__(self, input_shape, max_batch_size, model_output_shape=None):
        # Publics
        self.input_shape = input_shape
        self.max_batch_size = max_batch_size
        self.model_output_shape = model_output_shape

    def compute_halo(self, model_output_shape=None):
        model_output_shape = model_output_shape or self.model_output_shape
        assert_(self.model_output_shape is not None,
                "Cannot compute halo if model output shape is not set.",
                ValueError)
        # TODO
        return None


class ModelHandler(object):
    def __init__(self, *, model, device_names, in_channels, out_channels=1,
                 dynamic_shape_code):
        # Privates
        self._model = model
        self._max_batch_limit = 500
        # Publics
        self.device_names = to_list(device_names)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dynamic_shape = DynamicShape(dynamic_shape_code)

    @property
    def model(self):
        return self.model

    @property
    def device(self):
        return torch.device(self.device_names[0])

    @property
    def devices(self):
        return [torch.device(name) for name in self.device_names]

    def _trial_run_successful(self, *input_shape, device_id=None):
        if device_id is None:
            pass
            # TODO:
            # return [self._trial_run_successful(*input_shape, device_)]
        try:
            with torch.no_grad():
                self.model.to(self.devices[device_id])(torch.zeros(*input_shape))
            return True
        except torch.cuda.CudaError:
            # Nope
            return False

    def dry_run(self):
        pass

    def forward(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        # Check shape consistency

        pass
