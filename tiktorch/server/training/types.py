import enum


@enum.unique
class State(enum.Enum):
    Idle = "idle"
    Paused = "paused"
    Running = "running"
    Stopped = "stopped"
