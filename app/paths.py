from pathlib import Path

try:
    from .bootstrap import runtime_dirs
except ImportError:
    from bootstrap import runtime_dirs


def project_root():
    _base_dir, root_dir = runtime_dirs(__file__)
    return Path(root_dir)


def default_record_dir():
    return str(project_root() / "record")


def default_checkpoint_dir():
    return str(project_root() / "checkpoint")


def default_onnx_model_dir():
    return str(project_root() / "artifacts")
