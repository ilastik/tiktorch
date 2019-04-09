#############
# version 0.0
#############
TRAINING = "training"
VALIDATION = "validation"

# general
NAME = "name"
TORCH_VERSION = "torch_version"
INPUT_CHANNELS = "input_channels"
MODEL_CLASS_NAME = "model_class_name"
MODEL_INIT_KWARGS = "model_init_kwargs"
INPUT_AXIS_ORDER = "input_axis_order"
OUTPUT_AXIS_ORDER = "output_axis_order"
HALO = "halo"

# inference
INFERENCE_BATCH_SIZE = "inference_batch_size"

# training
BATCH_SIZE = "batch_size"
TRAINING_SHAPE = "training_shape"  # in tzyx, zyx, or yx (without batch_size and input_channels)
TRAINING_SHAPE_LOWER_BOUND = "training_shape_lower_bound"  # same conventions as for TRAINING_SHAPE
TRAINING_SHAPE_UPPER_BOUND = "training_shape_upper_bound"  # same conventions as for TRAINING_SHAPE
NUM_ITERATION_DONE = "num_iterations_done"
MAX_NUM_ITERATIONS = "max_num_iterations"
MAX_NUM_ITERATIONS_PER_UPDATE = "max_num_iterations_per_update"
LOSS_CRITERION_CONFIG = "loss_criterion_config"
OPTIMIZER_CONFIG = "optimizer_config"

# structure of config
MINIMAL_CONFIG = {
    MODEL_CLASS_NAME: None,
    INPUT_CHANNELS: None,
    INPUT_AXIS_ORDER: None,
    OUTPUT_AXIS_ORDER: None,
    TRAINING: {BATCH_SIZE: None, LOSS_CRITERION_CONFIG: None, OPTIMIZER_CONFIG: None},
    VALIDATION: {},
}

CONFIG = {
    NAME: None,
    TORCH_VERSION: None,
    MODEL_CLASS_NAME: None,
    MODEL_INIT_KWARGS: None,
    INPUT_CHANNELS: None,
    INPUT_AXIS_ORDER: None,
    OUTPUT_AXIS_ORDER: None,
    HALO: None,
    INFERENCE_BATCH_SIZE: None,
    TRAINING: {
        BATCH_SIZE: None,
        TRAINING_SHAPE: None,
        TRAINING_SHAPE_LOWER_BOUND: None,
        TRAINING_SHAPE_UPPER_BOUND: None,
        NUM_ITERATION_DONE: None,
        MAX_NUM_ITERATIONS: None,
        MAX_NUM_ITERATIONS_PER_UPDATE: None,
        LOSS_CRITERION_CONFIG: None,
        OPTIMIZER_CONFIG: None,
    },
    VALIDATION: {},
}
