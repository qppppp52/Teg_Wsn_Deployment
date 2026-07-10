#!/usr/bin/env python
"""Check project text files for UTF-8 BOM and CRLF/CR line endings."""
from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt"}
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "test_outputs", ".conda"}


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = []
    for path in iter_text_files(root):
        data = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        if data.startswith(bytes([0xEF, 0xBB, 0xBF])):
            errors.append(f"BOM: {rel}")
        if 13 in data:
            errors.append(f"CRLF_OR_CR: {rel}")
    if errors:
        print("Text encoding check failed:")
        for error in errors:
            print(error)
        return 1
    print("Text encoding check passed: UTF-8 without BOM and LF line endings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
