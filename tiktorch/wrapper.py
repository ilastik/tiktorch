import torch
import numpy as np

import logging


logger = logging.getLogger('TikTorch')


class TikTorch(object):
    """Wraps a torch model for use with LazyFlow and Ilastik."""
    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : torch.nn.Model
            Torch model.
        """
        assert isinstance(model, torch.nn.Module), \
            "Object must be a subclass of torch.nn.module."
        self._model = None
        self._configuration = {}
        self._preprocessors = None
        # Setter does the validation
        self.model = model
        if model is not None:
            logger.debug("Initialized with model. On GPU?: {}.".format(self.is_cuda))
        else:
            logger.debug("Initialized without model")

    @property
    def model(self):
        assert self._model is not None
        return self._model

    @model.setter
    def model(self, value):
        assert isinstance(value, torch.nn.Module)
        self._model = value.float()

    def bind_model(self, model):
        self.model = model
        return self

    def get(self, key, default=None):
        return self._configuration.get(key, default)

    def set(self, key, value):
        self._configuration.update({key: value})
        return self

    def increment(self, key):
        try:
            if key in self._configuration and isinstance(self._configuration[key], int):
                self._configuration[key] += 1
        except TypeError:
            pass

    def assert_defined(self, key, allow_none=False):
        key_in_configuration = key in self._configuration
        key_not_none = key_in_configuration and self._configuration[key] is not None
        assert key_in_configuration, \
            "Key {} not found in configuration.".format(key)
        if not allow_none:
            assert key_not_none, \
                "Key {} is defined in configuration, but maps to a None.".format(key)

    @property
    def is_cuda(self):
        try:
            return next(self.model.parameters()).is_cuda
        except StopIteration:
            # FIXME: Assuming that if a network has no parameters, it doesn't use CUDA
            return False

    def cuda(self, *args):
        """Transfers model to the GPU. Arguments specify the ids of the GPU devices to use."""
        self.model.cuda()
        if len(args) > 0:
            self.set('devices', list(args))
        return self

    def cpu(self):
        """Transfers model to the CPU."""
        self.model.cpu()
        return self

    def wrap_input_batch(self, input_batch):
        """Wraps numpy array as a torch variable on the right device."""
        # Convert to tensor
        assert isinstance(input_batch, np.ndarray)
        logger.debug("Converting input_batch to torch tensor.")
        input_batch_tensor = torch.from_numpy(input_batch).float()
        # Transfer to device
        if self.is_cuda:
            logger.debug("Transfering input_batch to GPU.")
            input_batch_tensor = input_batch_tensor.cuda()
        else:
            logger.debug("Using CPU.")
        logger.debug("Making variable from tensor.")
        # Make variable
        input_batch_variable = torch.autograd.Variable(input_batch_tensor,
                                                       volatile=True,
                                                       requires_grad=False)
        # Done
        return input_batch_variable

    def unwrap_output_batch(self, output_batch):
        """Unwraps torch variables to a numpy array."""
        assert isinstance(output_batch, torch.autograd.Variable)
        output_batch_tensor = output_batch.data
        # Transfer to CPU and convert to numpy array
        if self.is_cuda:
            logger.debug("Transferring output back to CPU from GPU.")
            output_batch_array = output_batch_tensor.cpu().numpy()
        else:
            logger.debug("Output is already on the CPU..")
            output_batch_array = output_batch_tensor.numpy()
        return output_batch_array

    def forward_through_model(self, *inputs):
        """
        Wrapper around the model's forward method. We might need this later for
        data-parallelism over multiple GPUs.
        """
        input_batch = inputs[0]
        # FIXME: Unhack
        # Normalize input batch
        logger.debug("Normalizing input batch.")
        logger.debug("Statistics before input normalization: mean = {}, min = {}, max = {}"
                     .format(input_batch.mean(), input_batch.min(), input_batch.max()))
        input_batch = (input_batch - input_batch.mean()) / (input_batch.std() + 0.000001)
        logger.debug("Statistics after input normalization: mean = {}, min = {}, max = {}, std = {}"
                     .format(input_batch.mean(),
                             input_batch.min(),
                             input_batch.max(),
                             input_batch.std()))
        logger.debug("Wrapping input_batch.")
        input_variable = self.wrap_input_batch(input_batch)
        # TODO Multi-GPU stuff goes here:
        logger.debug("Forward through model.")
        if self.get('devices') is None or not self.is_cuda:
            logger.debug("No devices specified or model is on the CPU.")
            output_variable = self.model(input_variable)
        else:
            logger.debug("Using devices: {}.".format(self.get('devices')))
            output_variable = torch.nn.parallel.data_parallel(self.model,
                                                              inputs=input_variable,
                                                              device_ids=self.get('devices'))
        logger.debug("Unwrapping output_variable.")
        output_batch = self.unwrap_output_batch(output_variable)
        logger.debug("Unwrapped output_variable.")
        logger.debug("Statistics of output array: mean = {}, min = {}, max = {}"
                     .format(input_batch.mean(), input_batch.min(), input_batch.max()))
        return output_batch

    @property
    def expected_input_shape(self):
        """Gets the input shape as expected from Lazyflow."""
        return (self.get('num_input_channels'),) + tuple(self.get('window_size'))

    @property
    def expected_output_shape(self):
        """Gets the output shape to be expected by Lazyflow."""
        return (self.get('num_output_channels'),) + tuple(self.get('window_size'))

    def change_shape(self, batch):
        """Changes the shape when it gets smaller than the expected output shape."""
        print(batch.shape,"batch")
        zero_arr = np.zeros((batch.shape[0],) + self.expected_output_shape)
        border = int((self.expected_output_shape[1] - batch.shape[2])/2)
        zero_arr[:, :, border:(border+batch.shape[2]), border:(border+batch.shape[3])] = batch
        return zero_arr

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            List of input batches, i.e. len(inputs) == number of batches.
            In 3D:
                If the object was configured with (say) window_size = [3, 512, 512],
                and num_input_channels = 5, inputs must be a list of numpy arrays of shape
                [5, 3, 512, 512].
            In 2D:
               If the object was configured with window_size = [512, 512] and
               num_input_channels = 1, inputs must be a list of numpy arrays of shape
               [1, 512, 512].

        Returns
        -------
        list
            List of outputs. Note that len(outputs) == len(inputs), and the contents of outputs
            have the same shape as that of inputs.
        """
        # We need CZYX (in ilastik-speak), or CDHW (in NN-speak)
        # We have a batch of inputs
        assert isinstance(inputs, (list, tuple)), \
            "Was expecting a list or tuple as `inputs`, got {} instead."\
                .format(inputs.__class__.__name__)
        logger.debug("Received input list of length {}.".format(len(inputs)))
        # Convert to a single numpy array
        input_batch = np.array(inputs)
        logger.debug("Shape of the input batch: {}.".format(input_batch.shape))
        assert input_batch.shape[1:] == self.expected_input_shape, \
            "Was expecting an input of shape {}, got one of shape {} instead."\
                .format(self.expected_input_shape, input_batch.shape[1:])  
        logger.debug("Feeding through model.")
        # Torch magic goes here:
        output_batch = self.forward_through_model(input_batch)
        # We expect an output of the same shape (which can be cropped
        # according to halo downstream). We still leave it flexible enough.
        if output_batch.shape[1:] != self.expected_output_shape:
            output_batch = self.change_shape(output_batch)

        logger.debug("Received output batch of shape {} from model.".format(output_batch.shape))
        assert output_batch.shape[1:] == self.expected_output_shape, \
            "Was expecting an output of shape {}, got one of shape {} instead." \
                .format(self.expected_output_shape, output_batch.shape[1:])
        # Separate outputs to list of batches
        outputs = list(output_batch)
        logger.debug("Returning list of {} array(s).".format(len(outputs)))
        return outputs

    def configure(self, *, window_size=None, output_size=None, num_input_channels=None, num_output_channels=None,
                  serialize_to_path=None, devices=None):
        """
        Configure the object.

        Parameters
        ----------
        window_size : list
            A length specifying the spatial shape of the input. Must be a list of
            length 2 for 2D images, or one of length 3 for 3D volumes.

        output_size : list
            A length specifying the spatial shape of the ouput. Must be a list of
            length 2 for 2D images, or one of length 3 for 3D volumes.

        num_input_channels : int
            Number of input channels. Must be an int >= 1.

        num_output_channels : int
            Number of output channels (the num classes the net predicts). Must be an int >= 1.

        serialize_to_path : str
            Where to serialize to. Must be a valid path.

        devices : list
            List of devices to use. By default, TikTorch will only use GPU0 if CUDA is available.

        Returns
        -------
        TikTorch
            self.

        """
        self.set('window_size', window_size)
        self.set('output_size', output_size)
        self.set('num_input_channels', num_input_channels)
        self.set('num_output_channels', num_output_channels)
        self.set('serialize_to_path', serialize_to_path)
        self.set('devices', devices)
        # TODO What else do we need?
        return self

    def serialize(self, to_path=None):
        to_path = self.get('serialize_to_path') if to_path is None else to_path
        assert to_path is not None, "Nowhere to serialize."
        torch.save(self, to_path)
        return self

    @classmethod
    def unserialize(cls, from_path):
        object_ = torch.load(from_path)
        assert isinstance(object_, cls), \
            "Object must be an instance of {}, " \
            "got an instance of {} instead.".format(cls.__name__,
                                                    object_.__class__.__name__)
        return object_

