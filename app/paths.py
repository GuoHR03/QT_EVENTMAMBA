import os
import sys
from pathlib import Path

from .bootstrap import runtime_dirs


def project_root():
    _base_dir, root_dir = runtime_dirs(__file__)
    return Path(root_dir)


def user_data_root(environ=None):
    environ = environ if environ is not None else os.environ
    preferred_root = environ.get("LOCALAPPDATA") or environ.get("APPDATA")
    if preferred_root:
        return Path(preferred_root) / "UI_Event"
    return Path.home() / "UI_Event"


def default_record_dir():
    if getattr(sys, "frozen", False):
        return str(user_data_root() / "record")
    return str(project_root() / "record")


def default_checkpoint_dir():
    return str(project_root() / "checkpoint")


def default_onnx_model_dir():
    return str(project_root() / "artifacts")
