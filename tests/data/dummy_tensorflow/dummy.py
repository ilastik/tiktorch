class TensorflowModelWrapper:
    def __init__(self):
        self._model = None

    def set_model(self, model):
        self._model = model

    def forward(self, input_):
        return self._model.predict(input_)

    def __call__(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)
