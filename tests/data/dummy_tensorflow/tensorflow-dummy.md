# Simple tensorflow test model

## Code to generate the model

```python
# Simple tensorflow test model
# inverts input.

## Code to generate the model

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda
import numpy
import xarray
import shutil

shutil.rmtree("./dummy-model")

model = Sequential()
model.add(Input(shape=(1, 32, 32), name="input0"))
model.add(Lambda(lambda x: x * -1, name="output0"))
tf.keras.experimental.export_saved_model(model, "./dummy-model")

input_data = numpy.random.randint(0, 255, (1, 1, 32, 32), dtype="uint8")
input_ = xarray.DataArray(input_data, dims=("b", "c", "y", "x"))

output_ = model.predict(input_)

numpy.save("test_input.npy", input_)
numpy.save("test_output.npy", output_)
```


```Python
# generate a model zoo model out of it
from bioimageio.core.build_spec import build_model
import shutil

weight_file = "./dummy-model-weights.zip"
name = "dummy-keras"
input_axes = ["bcyx"]
output_axes = ["bcyx"]
zip_path = "./dummy-model-tensorflow.zip"


shutil.make_archive("dummy-model-weights", "zip", root_dir="dummy-model")


new_mod_raw = build_model(
    weight_uri=weight_file,
    test_inputs=["./test_input.npy"],
    test_outputs=["./test_output.npy"],
    input_axes=input_axes,
    output_axes=output_axes,
    output_path=zip_path,
    name=name,
    input_names=["input0"],
    description="simple model that increments input by 1",
    authors=[{"name": "ilastik-team"}],
    documentation="./tensorflow-dummy.md",
    tags=["testing"],
    cite={"text": "pass"},
    tensorflow_version="1.14",
    weight_type="tensorflow_saved_model_bundle",
    halo=[[0, 0, 0, 0]],
    output_offset=[[0, 0, 0, 0]],
    output_reference=["input0"],
    output_scale=[[1.0, 1.0, 1.0, 1.0]],
    output_names=["output0"],
)

```
