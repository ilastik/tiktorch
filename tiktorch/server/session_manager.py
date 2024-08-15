from __future__ import annotations

import threading
from collections import defaultdict
from logging import getLogger
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from tiktorch.rpc.mp import BioModelClient

logger = getLogger(__name__)

CloseCallback = Callable[[], None]


class Session:
    """
    session object has unique id
    Used for resource managent
    """

    def __init__(self, id_: str, bio_model_client: BioModelClient, manager: SessionManager) -> None:
        self.__id = id_
        self.__manager = manager
        self.__bio_model_client = bio_model_client

    @property
    def bio_model_client(self) -> BioModelClient:
        return self.__bio_model_client

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


class SessionManager:
    """
    Manages session lifecycle (create/close)
    """

    def create_session(self, bio_model_client: BioModelClient) -> Session:
        """
        Creates new session with unique id
        """
        with self.__lock:
            session_id = uuid4().hex
            session = Session(session_id, bio_model_client=bio_model_client, manager=self)
            self.__session_by_id[session_id] = session
            logger.info("Created session %s", session.id)
            return session

    def get(self, session_id: str) -> Optional[Session]:
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
        self.__session_by_id: Dict[str, Session] = {}
        self.__close_handlers_by_session_id: Dict[str, List[CloseCallback]] = defaultdict(list)

    def _on_close(self, session: Session, handler: CloseCallback):
        with self.__lock:
            logger.debug("Registered close handler %s for session %s", handler, session.id)
            self.__close_handlers_by_session_id[session.id].append(handler)
