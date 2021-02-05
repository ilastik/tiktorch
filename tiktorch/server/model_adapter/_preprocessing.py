def _zero_mean_unit_variance(tensor, eps=1.0e-6):
    mean, std = tensor.mean(), tensor.std()
    return (tensor - mean) / (std + 1.0e-6)


KNOWN_PREPROCESSING = {
    "zero_mean_unit_variance": _zero_mean_unit_variance,
}


def chain(*functions):
    def _chained_function(tensor):
        tensor = tensor
        for fn in functions:
            tensor = fn(tensor)

        return tensor

    return _chained_function


def make_preprocessing(preprocessing):
    """
    :param preprocessing: bioimage-io spec node
    """
    preprocessing_functions = []

    for preprocessing_step in preprocessing:
        fn = KNOWN_PREPROCESSING.get(preprocessing_step.name)
        if fn is None:
            raise NotImplementedError(f"Preprocessing {preprocessing_step.name}")

        preprocessing_functions.append(fn)

    return chain(*preprocessing_functions)
