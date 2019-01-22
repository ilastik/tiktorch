# tiktorch
Wrappers for Pytorch

## Development environment installation

```
conda env create -f environment.yml
```

## tiktorch specification

Models saved for tiktorch must follow this specification:

A folder containing the files

- `model.py`: python file that defines the model.
- `state.nn`: state dict of the model, as obtained by `torch.save(model.state_dict(), 'state.nn')`
- `tiktorch_config.yml`: yaml file with metadata

The config must contain the following keys:

- `input_shape`: shape of valid network input, must be either CHW (2D) or CDHW (3D)
- `output_shape`: shape of network output given input with `input_shape`; same format
- `dynamic_input_shape`: TODO explain
- `model_class_name`: name of the model class in `model.py`
- `model_init_kwargs`: keyword arguments to build model
- `torch_version`: torch version used to train this model

In addition, the config may contain the following keys:

- `description`: Description of the pre-trained model
- `data_source`: URL of the data used for pre-training

TODO explain how to generate with tiktorch.

Possible extensions:

- specification for training set-up
- specifiying additional python modules necessary to run the model
- load model saved via `torch.save(model, path)` instead of state dict
