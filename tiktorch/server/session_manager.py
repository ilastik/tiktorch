from __future__ import annotations

import threading
from collections import defaultdict
from logging import getLogger
from typing import Callable, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4

logger = getLogger(__name__)

CloseCallback = Callable[[], None]

SessionClient = TypeVar("SessionClient")


class Session(Generic[SessionClient]):
    """
    session object has unique id
    Used for resource managent
    """

    def __init__(self, id_: str, client: SessionClient, manager: SessionManager) -> None:
        self.__id = id_
        self.__manager = manager
        self.__client = client

    @property
    def client(self) -> SessionClient:
        return self.__client

    @property
    def id(self) -> str:
        """
        Returns unique id assigned to this session
        """
        return self.__id

    def on_close(self, handler: CloseCallback) -> None:
        """
        Register cleanup function to be called when session ends
        """
        self.__manager._on_close(self, handler)


class SessionManager(Generic[SessionClient]):
    """
    Manages session lifecycle (create/close)
    """

    def create_session(self, client: SessionClient) -> Session[SessionClient]:
        """
        Creates new session with unique id
        """
        with self.__lock:
            session_id = uuid4().hex
            session = Session(session_id, client=client, manager=self)
            self.__session_by_id[session_id] = session
            logger.info("Created session %s", session.id)
            return session

    def get(self, session_id: str) -> Optional[Session[SessionClient]]:
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

    def close_all_sessions(self):
        all_ids = tuple(self.__session_by_id.keys())
        for session_id in all_ids:
            self.close_session(session_id)

    def __init__(self) -> None:
        self.__lock = threading.Lock()
        self.__session_by_id: Dict[str, Session] = {}
        self.__close_handlers_by_session_id: Dict[str, List[CloseCallback]] = defaultdict(list)

    def _on_close(self, session: Session, handler: CloseCallback):
        with self.__lock:
            logger.debug("Registered close handler %s for session %s", handler, session.id)
            self.__close_handlers_by_session_id[session.id].append(handler)
