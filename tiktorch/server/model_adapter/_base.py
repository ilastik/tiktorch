import abc
from typing import Callable


class ModelAdapter(abc.ABC):
    @abc.abstractmethod
    def forward(self, input_tensor):
        ...

    @property
    @abc.abstractmethod
    def max_num_iterations(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def iteration_count(self) -> int:
        ...

    @abc.abstractmethod
    def set_break_callback(self, thunk: Callable[[], bool]) -> None:
        ...

    @abc.abstractmethod
    def set_max_num_iterations(self, val: int) -> None:
        ...
