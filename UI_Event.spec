# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import re
import sys


project_root = Path(SPECPATH)
app_dir = project_root / 'app'
libs_dir = project_root / 'libs'
python_prefix = Path(sys.base_prefix if hasattr(sys, 'base_prefix') and sys.base_prefix else sys.prefix)
conda_bin = python_prefix / 'Library' / 'bin'

if sys.version_info[:2] != (3, 8):
    raise SystemExit(
        'UI_Event release builds require Python 3.8 because the bundled '
        'Metavision extensions target CPython 3.8.'
    )


def optional_binary(name):
    path = conda_bin / name
    if path.exists():
        return [(str(path), '.')]
    return []


def required_metavision_binary(name):
    path = libs_dir / 'bin' / name
    if not path.is_file():
        raise SystemExit(f'Required Metavision release DLL is missing: {path}')
    return [(str(path), 'libs/bin')]


def require_metavision_extension(name):
    path = libs_dir / name
    if not path.is_file():
        raise SystemExit(f'Required CPython 3.8 Metavision extension is missing: {path}')


def is_conflicting_libexpat(entry):
    src = Path(entry[1])
    return src.name.lower() == 'libexpat.dll' and src.parent != conda_bin


def is_unsupported_metavision_binary(entry):
    source = Path(entry[1])
    try:
        source.resolve().relative_to(libs_dir.resolve())
    except ValueError:
        return False
    name = source.name.lower()
    return bool(
        re.search(r'_d(?:\.dll|\.cp\d+-win_amd64\.pyd)$', name)
        or re.search(r'\.cp39-win_amd64\.pyd$', name)
    )


metavision_extensions = (
    'metavision_hal_internal.cp38-win_amd64.pyd',
    'metavision_sdk_base_internal.cp38-win_amd64.pyd',
    'metavision_sdk_base_paths_internal.cp38-win_amd64.pyd',
    'metavision_sdk_core_internal.cp38-win_amd64.pyd',
    'metavision_sdk_cv_internal.cp38-win_amd64.pyd',
)
for extension in metavision_extensions:
    require_metavision_extension(extension)

metavision_event_io_modules = [
    f'metavision_core.event_io.{path.stem}'
    for path in sorted((libs_dir / 'metavision_core' / 'event_io').glob('*.py'))
    if path.stem != '__init__'
]


a = Analysis(
    [str(project_root / 'main.py')],
    pathex=[str(project_root), str(app_dir), str(libs_dir)],
    binaries=(
        optional_binary('ffi.dll')
        + optional_binary('libbz2.dll')
        + optional_binary('LIBBZ2.dll')
        + optional_binary('libmpdec-4.dll')
        + optional_binary('libcrypto-3-x64.dll')
        + optional_binary('libssl-3-x64.dll')
        + optional_binary('sqlite3.dll')
        + optional_binary('libzmq-mt-4_3_5.dll')
        + optional_binary('expat.dll')
        + optional_binary('libexpat.dll')
        + required_metavision_binary('hdf5_ecf_codec.dll')
        + required_metavision_binary('metavision_hal.dll')
        + required_metavision_binary('metavision_hal_discovery.dll')
        + required_metavision_binary('metavision_sdk_base.dll')
        + required_metavision_binary('metavision_sdk_core.dll')
        + required_metavision_binary('metavision_sdk_cv.dll')
    ),
    datas=[
        (str(app_dir / 'form.ui'), 'app'),
    ],
    hiddenimports=[
        'metavision_core.event_io',
        'metavision_hal',
        'metavision_hal_internal',
        'metavision_sdk_base',
        'metavision_sdk_base_internal',
        'metavision_sdk_base_paths_internal',
        'metavision_sdk_core',
        'metavision_sdk_core_internal',
        'metavision_sdk_cv',
        'metavision_sdk_cv_internal',
        'dv_processing',
        'h5py',
    ] + metavision_event_io_modules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5',
        'PySide2',
        'PySide6',
        'IPython',
        'matplotlib',
        'pytest',
        'sphinx',
        'tkinter',
    ],
    noarchive=False,
    optimize=0,
)
a.binaries = [
    entry
    for entry in a.binaries
    if not is_conflicting_libexpat(entry)
    and not is_unsupported_metavision_binary(entry)
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='UI_Event',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='UI_Event',
)
