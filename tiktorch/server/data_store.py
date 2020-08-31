import abc
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class IDataStore(abc.ABC):
    @abc.abstractmethod
    def put(self, data: bytes) -> str:
        ...

    @abc.abstractmethod
    def get(self, id_: str) -> bytes:
        ...

    @abc.abstractmethod
    def remove(self, id_: str) -> None:
        ...


class DataStore(IDataStore):
    """
    Manages session lifecycle (create/close)
    """

    # TODO: Fix possible memory leakage
    # Options:
    # * Using temporary allocation for model upload
    # * Maybe attach to a session id, for this session should be created without
    # starting a model process

    def __init__(self):
        self.__data_by_id = {}

    def put(self, data: bytes) -> str:
        id_ = uuid4().hex
        self.__data_by_id[id_] = data
        return id_

    def get(self, id_: str) -> bytes:
        if id_ not in self.__data_by_id:
            raise Exception(f"Data blob with id {id_} not found")
        return self.__data_by_id[id_]

    def remove(self, id_: str):
        self.__data_by_id.pop(id_, None)
