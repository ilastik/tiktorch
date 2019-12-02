from uuid import uuid4


class _Session:
    def __init__(self, id_, *, _manager):
        self.__id = id_
        self.__manager = _manager

    @property
    def id(self):
        return self.__id

    def register_close_handler(self, handler):
        raise NotImplementedError()


class SessionManager:
    def __init__(self):
        self.__store = {}

    def create_session(self):
        session_id = uuid4().hex
        session = _Session(session_id, _manager=self)
        self.__store[session_id] = session
        return session

    def get(self, session_id):
        return self.__store.get(session_id, None)

    def close_session(self, session):
        raise NotImplementedError()
