#############
# version 0.0
#############
TRAINING = "training"
VALIDATION = "validation"

INPUT_CHANNELS = "input_channels"
MODEL_CLASS_NAME = "model_class_name"
MODEL_INIT_KWARGS = "model_init_kwargs"

# training
BATCH_SIZE = "batch_size"
TRAINING_SHAPE = "training_shape"  # in tzyx, zyx, or yx (without batch_size and input_channels)
TRAINING_SHAPE_UPPER_BOUND = "training_shape_upper_bound"  # same conventions as for TRAINING_SHAPE
TRAINING_SHAPE_LOWER_BOUND = "training_shape_lower_bound"  # same conventions as for TRAINING_SHAPE
NUM_ITERATION_DONE = "num_iterations_done"
MAX_NUM_ITERATIONS = "max_num_iterations"
MAX_NUM_ITERATIONS_PER_UPDATE = "max_num_iterations_per_update"
LOSS_CRITERION_CONFIG = "loss_criterion_config"
OPTIMIZER_CONFIG = "optimizer_config"

# structure of config
CONFIG = {
    MODEL_CLASS_NAME: None,
    MODEL_INIT_KWARGS: None,
    INPUT_CHANNELS: None,
    TRAINING: {BATCH_SIZE: None, TRAINING_SHAPE_LOWER_BOUND: None, TRAINING_SHAPE_UPPER_BOUND: None},
    VALIDATION: {},
}
