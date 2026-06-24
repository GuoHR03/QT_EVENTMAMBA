# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys


project_root = Path(SPECPATH)
app_dir = project_root / 'app'
python_prefix = Path(sys.base_prefix if hasattr(sys, 'base_prefix') and sys.base_prefix else sys.prefix)
conda_bin = python_prefix / 'Library' / 'bin'


def optional_binary(name):
    path = conda_bin / name
    if path.exists():
        return [(str(path), '.')]
    return []


def is_conflicting_libexpat(entry):
    src = Path(entry[1])
    return src.name.lower() == 'libexpat.dll' and src.parent != conda_bin


a = Analysis(
    [str(app_dir / 'widget.py')],
    pathex=[str(project_root), str(app_dir)],
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
    ),
    datas=[
        (str(app_dir / 'form.ui'), 'app'),
        (str(app_dir / 'choose_form.ui'), 'app'),
        (str(project_root / 'backend'), 'backend'),
        (str(project_root / 'libs'), 'libs'),
        (str(project_root / 'linux_backend.py'), '.'),
    ],
    hiddenimports=[],
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
a.binaries = [entry for entry in a.binaries if not is_conflicting_libexpat(entry)]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
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
