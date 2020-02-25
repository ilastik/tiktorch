#############
# version 0.0
#############
TRAINING = "session"
VALIDATION = "validation"
TESTING = "testing"
DRY_RUN = "dry_run"
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

# session
BATCH_SIZE = "batch_size"
TRAINING_SHAPE = "training_shape"  # e.g.  'tczyx', 'zyxc', or 'cxy' (without batch_size)
TRAINING_SHAPE_LOWER_BOUND = "training_shape_lower_bound"  # same conventions as for TRAINING_SHAPE
TRAINING_SHAPE_UPPER_BOUND = "training_shape_upper_bound"  # same conventions as for TRAINING_SHAPE
NUM_ITERATIONS_DONE = "num_iterations_done"
NUM_ITERATIONS_MAX = "num_iterations_max"
NUM_ITERATIONS_PER_UPDATE = "num_iterations_per_update"
LOSS_CRITERION_CONFIG = "loss_criterion_config"
OPTIMIZER_CONFIG = "optimizer_config"
TRAINING_LOSS = "training_loss"

# dryrun
SKIP = "skip"
SHRINKAGE = "shrinkage"

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
        NUM_ITERATIONS_DONE: None,
        NUM_ITERATIONS_MAX: None,
        NUM_ITERATIONS_PER_UPDATE: None,
        LOSS_CRITERION_CONFIG: None,
        OPTIMIZER_CONFIG: None,
        TRANSFORMS: None,
    },
    VALIDATION: {TRANSFORMS: None},
    TESTING: {TRANSFORMS: None},
    DRY_RUN: {SKIP: None, SHRINKAGE: None},
    LOGGING: {DIRECTORY: None},
}
