import hashlib
import json

import pytest

from tools.validate_asset_manifest import validate_asset_manifest


def _write_manifest(tmp_path, payload, digest=None):
    asset = tmp_path / "artifacts" / "model.onnx"
    asset.parent.mkdir()
    asset.write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "asset_set": "test",
        "runtime_assets": [
            {
                "path": "artifacts/model.onnx",
                "bytes": len(payload),
                "sha256": digest or hashlib.sha256(payload).hexdigest(),
                "role": "test model",
            }
        ],
        "source_assets": [],
    }
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_runtime_asset_manifest_verifies_size_and_hash(tmp_path):
    manifest_path = _write_manifest(tmp_path, b"model")

    result = validate_asset_manifest(manifest_path, tmp_path, scope="runtime")

    assert result["status"] == "verified"
    assert result["assets"] == ["artifacts/model.onnx"]


def test_runtime_asset_manifest_rejects_changed_asset(tmp_path):
    manifest_path = _write_manifest(tmp_path, b"model", digest="0" * 64)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_asset_manifest(manifest_path, tmp_path, scope="runtime")
