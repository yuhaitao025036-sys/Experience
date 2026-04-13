#!/usr/bin/env python3
"""
Convert Python source files to AST Tag JSONL format.

Usage:
    python convert_to_ast_tag.py <source_dir> <output_dir>
"""

import sys
import os
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from experience.ast_tag.convert_python_to_ast_tag_jsonl import convert_python_to_ast_tag_jsonl


def convert_directory(source_dir: str, output_dir: str) -> None:
    """Convert all .py files in source_dir to .jsonl ast_tag files."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for py_file in source_path.rglob("*.py"):
        rel_path = py_file.relative_to(source_path)
        jsonl_file = output_path / rel_path.with_suffix(".jsonl")
        jsonl_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(py_file, "r") as f:
                source = f.read()

            jsonl_output = convert_python_to_ast_tag_jsonl(source)

            with open(jsonl_file, "w") as f:
                f.write(jsonl_output)

            print(f"✓ {rel_path} -> {jsonl_file}")
            total_files += 1
        except Exception as e:
            print(f"✗ {rel_path}: {e}")

    print(f"\nConverted {total_files} files to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_ast_tag.py <source_dir> <output_dir>")
        sys.exit(1)

    convert_directory(sys.argv[1], sys.argv[2])
