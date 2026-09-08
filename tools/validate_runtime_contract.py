"""Validate the pinned interpreter and package contract for a runtime role."""

import argparse
import ctypes
import json
import platform
import sys
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "runtime-contract.json"


class RuntimeContractError(RuntimeError):
    pass


def load_contract(path=DEFAULT_CONTRACT):
    with Path(path).open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("schema_version") != 1:
        raise RuntimeContractError("Unsupported runtime contract schema")
    if not isinstance(contract.get("roles"), dict):
        raise RuntimeContractError("Runtime contract has no roles mapping")
    return contract


def windows_file_version(path):
    """Read a Windows executable's fixed file version without extra packages."""
    if sys.platform != "win32":
        raise RuntimeContractError("Windows file versions can only be read on Windows")

    version_api = ctypes.windll.version
    size = version_api.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        raise RuntimeContractError("No Windows version resource found: {}".format(path))

    buffer = ctypes.create_string_buffer(size)
    if not version_api.GetFileVersionInfoW(str(path), 0, size, buffer):
        raise RuntimeContractError("Could not read Windows version resource: {}".format(path))

    value = ctypes.c_void_p()
    value_length = ctypes.c_uint()
    if not version_api.VerQueryValueW(buffer, "\\", ctypes.byref(value), ctypes.byref(value_length)):
        raise RuntimeContractError("Could not query Windows file version: {}".format(path))

    words = ctypes.cast(value, ctypes.POINTER(ctypes.c_ushort))
    # VS_FIXEDFILEINFO: signature/structure version, followed by version MS/LS.
    file_version_ms = (words[5] << 16) | words[4]
    file_version_ls = (words[7] << 16) | words[6]
    parts = (
        file_version_ms >> 16,
        file_version_ms & 0xFFFF,
        file_version_ls >> 16,
        file_version_ls & 0xFFFF,
    )
    if parts[-1] == 0:
        parts = parts[:-1]
    return ".".join(str(part) for part in parts)


def validate_role(
    contract,
    role,
    python_version=None,
    distribution_version=metadata.version,
    sdk_root=None,
    file_version=windows_file_version,
):
    try:
        role_contract = contract["roles"][role]
    except KeyError as exc:
        raise RuntimeContractError("Unknown runtime role: {}".format(role)) from exc

    errors = []
    actual_python = python_version or platform.python_version()
    expected_python = role_contract["python"]
    if actual_python != expected_python:
        errors.append(
            "Python version mismatch: expected {}, got {}".format(
                expected_python,
                actual_python,
            )
        )

    actual_packages = {}
    for distribution, expected_version in role_contract.get("packages", {}).items():
        try:
            actual_version = distribution_version(distribution)
        except metadata.PackageNotFoundError:
            errors.append("Required distribution is missing: {}".format(distribution))
            continue
        actual_packages[distribution] = actual_version
        if actual_version != expected_version:
            errors.append(
                "{} version mismatch: expected {}, got {}".format(
                    distribution,
                    expected_version,
                    actual_version,
                )
            )

    sdk_report = None
    sdk_contract = role_contract.get("metavision")
    if sdk_contract is not None:
        if sdk_root is None:
            errors.append("Metavision SDK root is required for the UI contract")
        else:
            sdk_root = Path(sdk_root).resolve()
            version_path = sdk_root / Path(sdk_contract["version_file"])
            if not version_path.is_file():
                errors.append("Metavision version file is missing: {}".format(version_path))
            else:
                actual_sdk_version = file_version(version_path)
                expected_sdk_version = sdk_contract["version"]
                sdk_report = {
                    "root": str(sdk_root),
                    "version": actual_sdk_version,
                    "python_abi": sdk_contract["python_abi"],
                }
                if actual_sdk_version != expected_sdk_version:
                    errors.append(
                        "Metavision SDK version mismatch: expected {}, got {}".format(
                            expected_sdk_version,
                            actual_sdk_version,
                        )
                    )
            for extension in sdk_contract.get("required_extensions", []):
                extension_path = sdk_root / extension
                if not extension_path.is_file():
                    errors.append(
                        "Required Metavision extension is missing: {}".format(extension_path)
                    )

    if errors:
        raise RuntimeContractError("\n".join(errors))

    return {
        "status": "verified",
        "role": role,
        "python": actual_python,
        "packages": actual_packages,
        "metavision": sdk_report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument(
        "--role",
        choices=("ui", "windows_backend", "test"),
        required=True,
    )
    parser.add_argument("--sdk-root")
    args = parser.parse_args()

    try:
        result = validate_role(
            load_contract(args.contract),
            args.role,
            sdk_root=args.sdk_root,
        )
    except RuntimeContractError as exc:
        print("Runtime contract validation failed:\n{}".format(exc), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
