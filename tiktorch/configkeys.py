#############
# version 0.0
#############
TRAINING = "training"
VALIDATION = "validation"
TESTING = "testing"
LOGGING = "logging"

# general
NAME = "name"
TORCH_VERSION = "torch_version"
MODEL_CLASS_NAME = "model_class_name"
MODEL_INIT_KWARGS = "model_init_kwargs"
HALO = "halo"
TRANSFORMS = "transforms"

# inference
INFERENCE_BATCH_SIZE = "inference_batch_size"

# training
BATCH_SIZE = "batch_size"
TRAINING_SHAPE = "training_shape"  # e.g.  'tczyx', 'zyxc', or 'cxy' (without batch_size)
TRAINING_SHAPE_LOWER_BOUND = "training_shape_lower_bound"  # same conventions as for TRAINING_SHAPE
TRAINING_SHAPE_UPPER_BOUND = "training_shape_upper_bound"  # same conventions as for TRAINING_SHAPE
NUM_ITERATION_DONE = "num_iterations_done"
MAX_NUM_ITERATIONS = "max_num_iterations"
MAX_NUM_ITERATIONS_PER_UPDATE = "max_num_iterations_per_update"
LOSS_CRITERION_CONFIG = "loss_criterion_config"
OPTIMIZER_CONFIG = "optimizer_config"
TRAINING_LOSS = "training_loss"

# logging
DIRECTORY = "directory"


# structure of config
MINIMAL_CONFIG = {
    MODEL_CLASS_NAME: None,
    TRAINING: {BATCH_SIZE: None, LOSS_CRITERION_CONFIG: None, OPTIMIZER_CONFIG: None},
    VALIDATION: {},
}

CONFIG = {
    NAME: None,
    TORCH_VERSION: None,
    MODEL_CLASS_NAME: None,
    MODEL_INIT_KWARGS: None,
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
        TRANSFORMS: None,
    },
    VALIDATION: {TRANSFORMS: None},
    TESTING: {TRANSFORMS: None},
    LOGGING: {DIRECTORY: None},
}
