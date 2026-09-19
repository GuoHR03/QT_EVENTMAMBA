"""Validate first-party text files before tests and release builds."""

import argparse
from pathlib import Path


TEXT_SUFFIXES = {
    ".bat",
    ".cmd",
    ".cpp",
    ".cu",
    ".h",
    ".iss",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".spec",
    ".toml",
    ".ui",
    ".yml",
    ".yaml",
}
SOURCE_DIRECTORIES = (
    ".github",
    "app",
    "backend",
    "native",
    "requirements",
    "scripts",
    "tests",
    "tools",
)


def source_text_paths(project_root):
    project_root = Path(project_root)
    paths = [
        path
        for path in project_root.iterdir()
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]
    for directory_name in SOURCE_DIRECTORIES:
        directory = project_root / directory_name
        if not directory.is_dir():
            continue
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
        )
    return sorted(set(paths))


def validate_source_text(project_root):
    errors = []
    for path in source_text_paths(project_root):
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"{path}: invalid UTF-8 ({exc})")
            continue
        if "\ufffd" in content:
            errors.append(f"{path}: contains Unicode replacement character U+FFFD")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    errors = validate_source_text(args.project_root)
    if errors:
        raise SystemExit("Source text validation failed:\n" + "\n".join(errors))
    print("Source text validation passed.")


if __name__ == "__main__":
    main()
