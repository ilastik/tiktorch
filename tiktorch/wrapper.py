import torch
import numpy as np


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
        self._model = model
        self._configuration = {}

    @property
    def model(self):
        assert self._model is not None
        return self._model

    def get(self, key, default=None):
        return self._configuration.get(key, default)

    def set(self, key, value):
        self._configuration.update({key: value})
        return self

    @property
    def expected_input_shape(self):
        return (self.get('num_input_channels'),) + tuple(self.get('window_size'))

    @property
    def expected_output_shape(self):
        return self.expected_input_shape

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
        # Convert to a single numpy array
        input_batch = np.array(inputs)
        assert input_batch.shape[1:] == self.expected_input_shape, \
            "Was expecting an input of shape {}, got one of shape {} instead."\
                .format(self.expected_input_shape, input_batch.shape[1:])
        # Torch magic goes here:
        # ...
        # We expect an output of the same shape (which can be cropped
        # according to halo downstream). We still leave it flexible enough.
        output_batch = input_batch.copy()
        assert output_batch.shape[1:] == self.expected_output_shape, \
            "Was expecting an output of shape {}, got one of shape {} instead." \
                .format(self.expected_output_shape, output_batch.shape)
        # Separate outputs to list of batches
        outputs = list(output_batch)
        return outputs

    def configure(self, window_size=None, num_input_channels=None, serialize_to_path=None):
        """
        Configure the object.

        Parameters
        ----------
        window_size : list
            A length specifying the spatial shape of the input. Must be a list of
            length 2 for 2D images, or one of length 3 for 3D volumes.

        num_input_channels : int
            Number of input channels. Must be an int >= 1.

        serialize_to_path : str
            Where to serialize to. Must be a valid path.

        Returns
        -------
        TikTorch
            self.

        """
        self.set('window_size', window_size)
        self.set('num_input_channels', num_input_channels)
        self.set('serialize_to_path', serialize_to_path)
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

