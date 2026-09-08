import json
from importlib import metadata
from pathlib import Path

import pytest

from tools.validate_runtime_contract import RuntimeContractError, load_contract, validate_role


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _versions_for(role_contract):
    return dict(role_contract["packages"])


def _lookup(versions):
    def distribution_version(name):
        try:
            return versions[name]
        except KeyError as exc:
            raise metadata.PackageNotFoundError(name) from exc

    return distribution_version


def _requirement_pins(path):
    pins = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        pins[name] = version
    return pins


def test_repository_contract_defines_verified_runtime_and_test_roles():
    contract = load_contract()

    assert contract["roles"]["ui"]["python"] == "3.8.20"
    assert contract["roles"]["ui"]["metavision"]["version"] == "4.6.2"
    assert contract["roles"]["windows_backend"]["python"] == "3.13.5"
    assert contract["roles"]["windows_backend"]["packages"]["onnxruntime-gpu"] == "1.27.0"
    assert contract["roles"]["test"]["python"] == "3.13.5"
    assert contract["roles"]["test"]["packages"]["pytest"] == "8.3.4"


@pytest.mark.parametrize(
    ("role", "requirements_file"),
    (
        ("ui", "ui-py38.txt"),
        ("windows_backend", "backend-py313.txt"),
        ("test", "test-py313.txt"),
    ),
)
def test_requirement_files_match_all_published_contract_pins(role, requirements_file):
    contract = load_contract()
    contract_packages = contract["roles"][role]["packages"]
    requirement_pins = _requirement_pins(PROJECT_ROOT / "requirements" / requirements_file)

    for package, expected_version in requirement_pins.items():
        assert contract_packages[package] == expected_version


def test_backend_contract_accepts_exact_interpreter_and_package_versions():
    contract = load_contract()
    role_contract = contract["roles"]["windows_backend"]

    result = validate_role(
        contract,
        "windows_backend",
        python_version=role_contract["python"],
        distribution_version=_lookup(_versions_for(role_contract)),
    )

    assert result["status"] == "verified"


def test_contract_reports_python_and_package_drift_together():
    contract = load_contract()
    role_contract = contract["roles"]["windows_backend"]
    versions = _versions_for(role_contract)
    versions["onnxruntime-gpu"] = "0.0.0"

    with pytest.raises(RuntimeContractError) as error:
        validate_role(
            contract,
            "windows_backend",
            python_version="3.12.0",
            distribution_version=_lookup(versions),
        )

    assert "Python version mismatch" in str(error.value)
    assert "onnxruntime-gpu version mismatch" in str(error.value)


def test_ui_contract_checks_metavision_version_and_cp38_extensions(tmp_path):
    contract = load_contract()
    role_contract = contract["roles"]["ui"]
    sdk_contract = role_contract["metavision"]
    version_file = tmp_path / Path(sdk_contract["version_file"])
    version_file.parent.mkdir(parents=True)
    version_file.touch()
    for extension in sdk_contract["required_extensions"]:
        (tmp_path / extension).touch()

    result = validate_role(
        contract,
        "ui",
        python_version=role_contract["python"],
        distribution_version=_lookup(_versions_for(role_contract)),
        sdk_root=tmp_path,
        file_version=lambda _path: "4.6.2",
    )

    assert result["metavision"]["python_abi"] == "cp38"
