from typing import List, Tuple

SHUTDOWN: Tuple[str, dict] = ("shutdown", {})
SHUTDOWN_ANSWER: Tuple[str, dict] = ("shutting_down", {})
REPORT_EXCEPTION: str = "report_exception"
TRAINING: str = "train"  # same name as used in inferno
VALIDATION: str = "validate"  # same name as used in inferno
REQUEST_FOR_DEVICES: List = ['request']
