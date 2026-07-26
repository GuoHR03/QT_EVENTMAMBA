import os
import sys


def to_wsl_path(path, wsl_distro="EventMamba_mini"):
    if not path:
        return path

    if path.startswith("\\\\wsl$\\"):
        parts = path.split("\\")
        if len(parts) >= 5:
            distro = parts[3]
            inner = "/".join(parts[4:])
            if distro == wsl_distro:
                return f"/{inner}".replace("\\", "/")
            return f"/mnt/wsl/{distro}/{inner}".replace("\\", "/")

    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"

    return path


def runtime_resource_dir(module_file, frozen=None, meipass=None):
    if frozen is None:
        frozen = getattr(sys, "frozen", False)
    if meipass is None:
        meipass = getattr(sys, "_MEIPASS", None)
    if frozen and meipass:
        return os.path.abspath(meipass)

    current_dir = os.path.dirname(os.path.abspath(module_file))
    return os.path.dirname(current_dir)


def runtime_root_dir(module_file, frozen=None, meipass=None, executable=None):
    """Return the stable root used for installed runtime files."""
    if frozen is None:
        frozen = getattr(sys, "frozen", False)
    if frozen:
        executable = executable or sys.executable
        if executable:
            return os.path.dirname(os.path.abspath(executable))
        if meipass is None:
            meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return os.path.abspath(meipass)

    current_dir = os.path.dirname(os.path.abspath(module_file))
    return os.path.dirname(current_dir)


def default_backend_log_path(runtime_root, environ=None):
    environ = environ if environ is not None else os.environ
    preferred_root = environ.get("LOCALAPPDATA") or environ.get("TEMP") or runtime_root
    log_dir = os.path.join(preferred_root, "UI_Event")
    return os.path.join(log_dir, "eventmamba_backend.log")


def decode_backend_log(raw):
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass

    for encoding in ("utf-8", "gbk", "utf-16-le"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_float_env(name, default, environ=None):
    environ = environ if environ is not None else os.environ
    try:
        return float(environ.get(name, default))
    except (TypeError, ValueError):
        return default
