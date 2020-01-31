from __future__ import annotations

import abc
import threading

from typing import Callable, Dict, List, Optional
from uuid import uuid4
from collections import defaultdict
from logging import getLogger

logger = getLogger(__name__)


class IModel(abc.ABC):
    """
    model object has unique id
    Used for resource managent
    """

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """
        Returns unique id assigned to this model
        """
        ...

    @abc.abstractmethod
    def on_close(self, handler: CloseCallback) -> None:
        """
        Register cleanup function to be called when model ends
        """
        ...


CloseCallback = Callable[[IModel], None]


class _Model(IModel):
    def __init__(self, id_: str, manager: ModelManager) -> None:
        self.__id = id_
        self.__manager = manager

    @property
    def id(self) -> str:
        return self.__id

    def on_close(self, handler: CloseCallback) -> None:
        self.__manager._on_close(self, handler)


class ModelManager:
    """
    Manages model lifecycle (create/close)
    """

    def create_model(self) -> IModel:
        """
        Creates new model with unique id
        """
        with self.__lock:
            model_id = uuid4().hex
            model = _Model(model_id, manager=self)
            self.__model_by_id[model_id] = model
            logger.info("Created model %s", model.id)
            return model

    def get(self, model_id: str) -> Optional[IModel]:
        """
        Returns existing model with given id if it exists
        """
        with self.__lock:
            return self.__model_by_id.get(model_id, None)

    def close_model(self, model_id: str) -> None:
        """
        Closes model with given id if it exists and invokes close handlers
        """
        with self.__lock:
            if model_id not in self.__model_by_id:
                raise ValueError("Unknown model")

            model = self.__model_by_id.pop(model_id)
            for handler in self.__close_handlers_by_model_id.pop(model_id, []):
                try:
                    handler()
                except Exception:
                    logger.exception("Error during close handler for model %s", model_id)

            logger.debug("Closed model %s", model_id)

    def __init__(self) -> None:
        self.__lock = threading.Lock()
        self.__model_by_id: Dict[str, IModel] = {}
        self.__close_handlers_by_model_id: Dict[str, List[CloseCallback]] = defaultdict(list)

    def _on_close(self, model: IModel, handler: CloseCallback):
        with self.__lock:
            logger.debug("Registered close handler %s for model %s", handler, model.id)
            self.__close_handlers_by_model_id[model.id].append(handler)
