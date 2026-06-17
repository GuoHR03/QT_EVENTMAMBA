import os
import sys


def runtime_dirs(module_file):
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(module_file)))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    return base_dir, root_dir


def configure_runtime(module_file):
    base_dir, root_dir = runtime_dirs(module_file)
    for path in (root_dir, base_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    sdk_root = os.environ.get("METAVISION_SDK_PATH", "E:\\Metavision\\Prophesee")
    extra_dll_dirs = [
        os.path.join(root_dir, "libs", "bin"),
        os.path.join(sdk_root, "bin"),
        os.path.join(sdk_root, "third_party", "bin"),
        os.path.join(sdk_root, "lib", "hdf5", "plugin"),
    ]
    if hasattr(os, "add_dll_directory"):
        for dll_dir in extra_dll_dirs:
            if os.path.isdir(dll_dir):
                os.add_dll_directory(dll_dir)
    return base_dir, root_dir


def app_resource_path(filename, module_file=__file__):
    base_dir, _root_dir = runtime_dirs(module_file)
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "app", filename),
        os.path.join(os.path.dirname(os.path.abspath(module_file)), filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]
