from __future__ import annotations

import abc
import threading
from collections import defaultdict
from logging import getLogger
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from tiktorch.server.session import IRPCModelSession

logger = getLogger(__name__)


class ISession(abc.ABC):
    """
    session object has unique id
    Used for resource managent
    """

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """
        Returns unique id assigned to this session
        """
        ...

    @property
    @abc.abstractmethod
    def client(self) -> IRPCModelSession:
        ...

    @abc.abstractmethod
    def on_close(self, handler: CloseCallback) -> None:
        """
        Register cleanup function to be called when session ends
        """
        ...


CloseCallback = Callable[[], None]


class _Session(ISession):
    def __init__(self, id_: str, client: IRPCModelSession, manager: SessionManager) -> None:
        self.__id = id_
        self.__manager = manager
        self.__client = client

    @property
    def client(self) -> IRPCModelSession:
        return self.__client

    @property
    def id(self) -> str:
        return self.__id

    def on_close(self, handler: CloseCallback) -> None:
        self.__manager._on_close(self, handler)


class SessionManager:
    """
    Manages session lifecycle (create/close)
    """

    def create_session(self, client: IRPCModelSession) -> ISession:
        """
        Creates new session with unique id
        """
        with self.__lock:
            session_id = uuid4().hex
            session = _Session(session_id, client=client, manager=self)
            self.__session_by_id[session_id] = session
            logger.info("Created session %s", session.id)
            return session

    def get(self, session_id: str) -> Optional[ISession]:
        """
        Returns existing session with given id if it exists
        """
        with self.__lock:
            return self.__session_by_id.get(session_id, None)

    def close_session(self, session_id: str) -> None:
        """
        Closes session with given id if it exists and invokes close handlers
        """
        with self.__lock:
            if session_id not in self.__session_by_id:
                raise ValueError("Unknown session")

            self.__session_by_id.pop(session_id)
            for handler in self.__close_handlers_by_session_id.pop(session_id, []):
                try:
                    handler()
                except Exception:
                    logger.exception("Error during close handler for session %s", session_id)

            logger.debug("Closed session %s", session_id)

    def __init__(self) -> None:
        self.__lock = threading.Lock()
        self.__session_by_id: Dict[str, ISession] = {}
        self.__close_handlers_by_session_id: Dict[str, List[CloseCallback]] = defaultdict(list)

    def _on_close(self, session: ISession, handler: CloseCallback):
        with self.__lock:
            logger.debug("Registered close handler %s for session %s", handler, session.id)
            self.__close_handlers_by_session_id[session.id].append(handler)
