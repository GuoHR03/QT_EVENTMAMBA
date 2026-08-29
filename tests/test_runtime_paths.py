import os
import sys

from app import bootstrap
from app import paths


def test_runtime_dirs_preserve_source_tree_layout(tmp_path):
    project_dir = tmp_path / "project"
    app_dir = project_dir / "app"
    module_file = app_dir / "widget.py"

    resource_root, install_root = bootstrap.runtime_dirs(
        str(module_file),
        frozen=False,
    )

    assert resource_root == str(app_dir)
    assert install_root == str(project_dir)


def test_runtime_dirs_split_frozen_resource_and_install_roots(tmp_path):
    bundle_dir = tmp_path / "bundle"
    install_dir = tmp_path / "UI_Event"

    resource_root, install_root = bootstrap.runtime_dirs(
        str(bundle_dir / "app" / "widget.py"),
        frozen=True,
        meipass=str(bundle_dir),
        executable=str(install_dir / "UI_Event.exe"),
    )

    assert resource_root == str(bundle_dir)
    assert install_root == str(install_dir)


def test_configure_runtime_registers_bundled_and_installed_libraries(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    install_dir = tmp_path / "UI_Event"
    for root in (bundle_dir, install_dir):
        (root / "libs" / "bin").mkdir(parents=True)
    bundled_sdk_dir = install_dir / "metavision"
    bundled_sdk_bin = bundled_sdk_dir / "bin"
    bundled_sdk_third_party = bundled_sdk_dir / "third_party" / "bin"
    bundled_hdf5_plugins = bundled_sdk_dir / "lib" / "hdf5" / "plugin"
    bundled_hal_plugins = (
        bundled_sdk_dir / "lib" / "metavision" / "hal" / "plugins"
    )
    for directory in (
        bundled_sdk_bin,
        bundled_sdk_third_party,
        bundled_hdf5_plugins,
        bundled_hal_plugins,
    ):
        directory.mkdir(parents=True)

    registered_dll_dirs = []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
    monkeypatch.setattr(sys, "executable", str(install_dir / "UI_Event.exe"))
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(
        bootstrap.os,
        "add_dll_directory",
        lambda path: registered_dll_dirs.append(path) or object(),
        raising=False,
    )
    monkeypatch.delenv("METAVISION_SDK_PATH", raising=False)
    existing_hdf5_plugin_dir = tmp_path / "user-hdf5-plugins"
    monkeypatch.setenv("HDF5_PLUGIN_PATH", str(existing_hdf5_plugin_dir))
    existing_hal_plugin_dir = tmp_path / "user-hal-plugins"
    monkeypatch.setenv("MV_HAL_PLUGIN_PATH", str(existing_hal_plugin_dir))

    assert bootstrap.configure_runtime(__file__) == (str(bundle_dir), str(install_dir))

    expected_library_dirs = [
        str(bundle_dir / "libs"),
        str(bundle_dir / "libs" / "bin"),
        str(install_dir / "libs"),
        str(install_dir / "libs" / "bin"),
    ]
    assert sys.path == [str(bundle_dir), str(install_dir)] + expected_library_dirs
    assert registered_dll_dirs == expected_library_dirs + [
        str(bundled_sdk_bin),
        str(bundled_sdk_third_party),
        str(bundled_hdf5_plugins),
        str(bundled_hal_plugins),
    ]
    assert os.environ["HDF5_PLUGIN_PATH"].split(os.pathsep) == [
        str(existing_hdf5_plugin_dir),
        str(bundled_hdf5_plugins),
    ]
    assert os.environ["MV_HAL_PLUGIN_PATH"].split(os.pathsep) == [
        str(existing_hal_plugin_dir),
        str(bundled_hal_plugins),
    ]


def test_configure_runtime_uses_source_tree_libs_as_sdk_runtime(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    app_dir = project_dir / "app"
    module_file = app_dir / "widget.py"
    libs_dir = project_dir / "libs"
    sdk_directories = (
        libs_dir / "bin",
        libs_dir / "third_party" / "bin",
        libs_dir / "lib" / "hdf5" / "plugin",
        libs_dir / "lib" / "metavision" / "hal" / "plugins",
    )
    for directory in sdk_directories:
        directory.mkdir(parents=True)

    registered_dll_dirs = []
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(
        bootstrap.os,
        "add_dll_directory",
        lambda path: registered_dll_dirs.append(path) or object(),
        raising=False,
    )
    monkeypatch.delenv("METAVISION_SDK_PATH", raising=False)
    monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
    monkeypatch.delenv("MV_HAL_PLUGIN_PATH", raising=False)

    assert bootstrap.configure_runtime(str(module_file)) == (
        str(app_dir),
        str(project_dir),
    )

    assert str(libs_dir) in sys.path
    assert str(libs_dir / "bin") in sys.path
    assert registered_dll_dirs == [str(libs_dir), *map(str, sdk_directories)]
    assert os.environ["HDF5_PLUGIN_PATH"] == str(sdk_directories[2])
    assert os.environ["MV_HAL_PLUGIN_PATH"] == str(sdk_directories[3])


def test_app_resource_path_prefers_bundled_app_resource(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    install_dir = tmp_path / "UI_Event"
    bundled_form = bundle_dir / "app" / "form.ui"
    bundled_form.parent.mkdir(parents=True)
    bundled_form.touch()

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
    monkeypatch.setattr(sys, "executable", str(install_dir / "UI_Event.exe"))

    assert bootstrap.app_resource_path("form.ui", module_file=__file__) == str(bundled_form)


def test_default_paths_use_source_project_root(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    monkeypatch.setattr(paths, "runtime_dirs", lambda _module: (str(project_dir / "app"), str(project_dir)))
    monkeypatch.setattr(paths.sys, "frozen", False, raising=False)

    assert paths.default_record_dir() == str(project_dir / "record")
    assert paths.default_checkpoint_dir() == str(project_dir / "checkpoint")
    assert paths.default_onnx_model_dir() == str(project_dir / "artifacts")


def test_frozen_record_dir_uses_local_app_data_but_assets_use_install_root(tmp_path, monkeypatch):
    install_dir = tmp_path / "UI_Event"
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.setattr(paths, "runtime_dirs", lambda _module: (str(tmp_path / "bundle"), str(install_dir)))
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    assert paths.default_record_dir() == str(local_app_data / "UI_Event" / "record")
    assert paths.default_checkpoint_dir() == str(install_dir / "checkpoint")
    assert paths.default_onnx_model_dir() == str(install_dir / "artifacts")


def test_user_data_root_falls_back_to_appdata(tmp_path):
    assert paths.user_data_root(
        {"APPDATA": str(tmp_path / "Roaming")}
    ) == tmp_path / "Roaming" / "UI_Event"
