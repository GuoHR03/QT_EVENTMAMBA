"""Verify release and source inference assets against their manifest."""

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "artifacts" / "manifest.json"
ASSET_GROUPS = ("runtime_assets", "source_assets")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_asset_manifest(manifest_path=DEFAULT_MANIFEST, project_root=PROJECT_ROOT, scope="all"):
    manifest_path = Path(manifest_path)
    project_root = Path(project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported asset manifest schema")

    groups = ASSET_GROUPS if scope == "all" else (f"{scope}_assets",)
    verified = []
    seen_paths = set()
    for group in groups:
        entries = manifest.get(group)
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"Asset manifest group is missing or empty: {group}")
        for entry in entries:
            relative_path = entry.get("path")
            if not isinstance(relative_path, str) or not relative_path:
                raise RuntimeError(f"Invalid asset path in {group}")
            normalized = Path(relative_path).as_posix()
            if normalized in seen_paths:
                raise RuntimeError(f"Duplicate asset path in manifest: {normalized}")
            seen_paths.add(normalized)

            path = project_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Asset is missing: {path}")
            expected_size = int(entry.get("bytes", -1))
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Asset size mismatch for {relative_path}: "
                    f"expected {expected_size}, got {actual_size}"
                )
            expected_hash = str(entry.get("sha256", "")).lower()
            actual_hash = file_sha256(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Asset SHA-256 mismatch for {relative_path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            verified.append(relative_path)
    return {
        "status": "verified",
        "asset_set": manifest.get("asset_set"),
        "scope": scope,
        "assets": verified,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--scope", choices=("all", "runtime", "source"), default="all")
    args = parser.parse_args(argv)
    result = validate_asset_manifest(args.manifest, args.project_root, args.scope)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
