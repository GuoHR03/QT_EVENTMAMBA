import os
import sys


_DLL_DIRECTORY_HANDLES = []


def runtime_dirs(module_file, frozen=None, meipass=None, executable=None):
    """Return the bundled-resource root and the stable installation root."""
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))

    if frozen:
        executable = executable or sys.executable
        install_root = os.path.dirname(os.path.abspath(executable))
        if meipass is None:
            meipass = getattr(sys, "_MEIPASS", None)
        resource_root = os.path.abspath(meipass) if meipass else install_root
        return resource_root, install_root

    resource_root = os.path.dirname(os.path.abspath(module_file))
    install_root = os.path.abspath(os.path.join(resource_root, ".."))
    return resource_root, install_root


def _unique_paths(paths):
    result = []
    seen = set()
    for path in paths:
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized not in seen:
            result.append(path)
            seen.add(normalized)
    return result


def _append_environment_paths(name, paths):
    current_paths = [
        path for path in os.environ.get(name, "").split(os.pathsep) if path
    ]
    combined_paths = _unique_paths(current_paths + list(paths))
    if combined_paths:
        os.environ[name] = os.pathsep.join(combined_paths)


def configure_runtime(module_file):
    resource_root, install_root = runtime_dirs(module_file)
    roots = _unique_paths((resource_root, install_root))
    library_dirs = _unique_paths(
        os.path.join(root, relative)
        for root in roots
        for relative in ("libs", os.path.join("libs", "bin"))
    )

    python_dirs = roots + [path for path in library_dirs if os.path.isdir(path)]
    for path in reversed(python_dirs):
        if path not in sys.path:
            sys.path.insert(0, path)

    bundled_sdk_roots = []
    for root in (install_root, resource_root):
        for directory_name in ("metavision", "libs"):
            candidate = os.path.join(root, directory_name)
            if all(
                os.path.isdir(os.path.join(candidate, relative))
                for relative in (
                    os.path.join("third_party", "bin"),
                    os.path.join("lib", "hdf5", "plugin"),
                    os.path.join("lib", "metavision", "hal", "plugins"),
                )
            ):
                bundled_sdk_roots.append(candidate)
    sdk_root = os.environ.get("METAVISION_SDK_PATH")
    if not sdk_root:
        sdk_root = (
            bundled_sdk_roots[0]
            if bundled_sdk_roots
            else "E:\\Metavision\\Prophesee"
        )

    hdf5_plugin_dir = os.path.join(sdk_root, "lib", "hdf5", "plugin")
    hal_plugin_dir = os.path.join(
        sdk_root,
        "lib",
        "metavision",
        "hal",
        "plugins",
    )
    _append_environment_paths(
        "HDF5_PLUGIN_PATH",
        [hdf5_plugin_dir] if os.path.isdir(hdf5_plugin_dir) else [],
    )
    _append_environment_paths(
        "MV_HAL_PLUGIN_PATH",
        [hal_plugin_dir] if os.path.isdir(hal_plugin_dir) else [],
    )

    extra_dll_dirs = library_dirs + [
        os.path.join(sdk_root, "bin"),
        os.path.join(sdk_root, "third_party", "bin"),
        hdf5_plugin_dir,
        hal_plugin_dir,
    ]
    if hasattr(os, "add_dll_directory"):
        for dll_dir in _unique_paths(extra_dll_dirs):
            if os.path.isdir(dll_dir):
                handle = os.add_dll_directory(dll_dir)
                _DLL_DIRECTORY_HANDLES.append(handle)
    return resource_root, install_root


def app_resource_path(filename, module_file=__file__):
    resource_root, install_root = runtime_dirs(module_file)
    candidates = [
        os.path.join(resource_root, filename),
        os.path.join(resource_root, "app", filename),
        os.path.join(install_root, filename),
        os.path.join(install_root, "app", filename),
        os.path.join(os.path.dirname(os.path.abspath(module_file)), filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]
