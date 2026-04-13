#!/usr/bin/env python3
"""
Interactive navigation tool for ast_tag database.

Usage:
    python navigate_db.py <dataset_dir>

Commands:
    refs <symbol>     - Find all references to symbol
    def <file:tag>   - Go to symbol definition
    parent <file:tag> - Find lexical parent scope
    children <file:tag> - Expand children of scope
    symbols <file>   - List all symbols in file
    lines <file>     - Show line counts per file
    quit             - Exit
"""

import sys
import os
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from experience.ast_tag.ast_tag_db import load_jsonl_dataset_into_ast_tag_db
from experience.ast_tag.tag_actions.dynamic_scope_find_all_references import dynamic_scope_find_all_references
from experience.ast_tag.tag_actions.dynamic_scope_go_to_definition import dynamic_scope_go_to_definition
from experience.ast_tag.tag_actions.lexical_scope_go_to_parent import lexical_scope_go_to_parent
from experience.ast_tag.tag_actions.lexical_scope_expand_children import lexical_scope_expand_children


def main():
    if len(sys.argv) != 2:
        print("Usage: python navigate_db.py <dataset_dir>")
        sys.exit(1)

    print(f"Loading ast_tag database from {sys.argv[1]}...")
    db = load_jsonl_dataset_into_ast_tag_db(sys.argv[1])

    file_ids = db.get_all_loaded_file_ids()
    print(f"Loaded {len(file_ids)} files.\n")

    while True:
        try:
            cmd = input("ast_tag> ").strip()
        except EOFError:
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()

        if action == "quit" or action == "exit" or action == "q":
            break

        elif action == "refs":
            if len(parts) < 2:
                print("Usage: refs <symbol>")
                continue
            symbol = parts[1]
            refs = dynamic_scope_find_all_references(db, symbol)
            print(f"References to '{symbol}': {len(refs)} found")
            for ref in refs[:10]:
                print(f"  {ref.file_id}:{ref.line} {ref.owner_tag} [{ref.relation_tag}] -> {ref.member_tag}")
            if len(refs) > 10:
                print(f"  ... and {len(refs) - 10} more")

        elif action == "def":
            if len(parts) < 2:
                print("Usage: def <file:tag>")
                continue
            try:
                file_tag = parts[1].split(":", 1)
                file_id = file_tag[0]
                member_tag = file_tag[1] if len(file_tag) > 1 else None
                if member_tag is None:
                    print("Usage: def <file:tag>")
                    continue
                defn = dynamic_scope_go_to_definition(db, file_id, member_tag)
                if defn:
                    print(f"Definition: {defn.file_id}:{defn.line} {defn.owner_tag} [{defn.relation_tag}]")
                else:
                    print("Definition not found")
            except Exception as e:
                print(f"Error: {e}")

        elif action == "parent":
            if len(parts) < 2:
                print("Usage: parent <file:tag>")
                continue
            try:
                file_tag = parts[1].split(":", 1)
                file_id = file_tag[0]
                member_tag = file_tag[1] if len(file_tag) > 1 else None
                if member_tag is None:
                    print("Usage: parent <file:tag>")
                    continue
                parent = lexical_scope_go_to_parent(db, file_id, member_tag)
                if parent:
                    print(f"Parent: {parent.file_id}:{parent.line} {parent.owner_tag} [{parent.relation_tag}] -> {parent.member_tag}")
                else:
                    print("Parent not found")
            except ValueError as e:
                print(f"Error: {e}")

        elif action == "children":
            if len(parts) < 2:
                print("Usage: children <file:tag>")
                continue
            try:
                file_tag = parts[1].split(":", 1)
                file_id = file_tag[0]
                owner_tag = file_tag[1] if len(file_tag) > 1 else None
                if owner_tag is None:
                    print("Usage: children <file:tag>")
                    continue
                children = lexical_scope_expand_children(db, file_id, owner_tag)
                print(f"Children of '{owner_tag}': {len(children)} found")
                for child in children[:15]:
                    print(f"  {child.line} {child.owner_tag} [{child.relation_tag}] -> {child.member_tag}")
                if len(children) > 15:
                    print(f"  ... and {len(children) - 15} more")
            except Exception as e:
                print(f"Error: {e}")

        elif action == "symbols":
            if len(parts) < 2:
                print("Usage: symbols <file>")
                continue
            file_id = parts[1]
            rows = db.execute_raw_sql_query(
                "SELECT DISTINCT owner_tag, relation_tag, member_tag FROM relations WHERE file_id = ?",
                (file_id,),
            )
            print(f"Symbols in '{file_id}': {len(rows)} found")
            for owner, rel, member in rows[:20]:
                print(f"  {owner} --[{rel}]--> {member}")
            if len(rows) > 20:
                print(f"  ... and {len(rows) - 20} more")

        elif action == "lines":
            print("Files and line counts:")
            for fid in file_ids[:20]:
                count = db.count_file_relation_records(fid)
                print(f"  {fid}: {count} records")
            if len(file_ids) > 20:
                print(f"  ... and {len(file_ids) - 20} more files")

        elif action == "help":
            print("""Commands:
  refs <symbol>      - Find all references to symbol
  def <file:tag>     - Go to symbol definition
  parent <file:tag>  - Find lexical parent scope
  children <file:tag> - Expand children of scope
  symbols <file>     - List all symbols in file
  lines              - Show line counts per file
  quit               - Exit
""")

        else:
            print(f"Unknown command: {action}. Type 'help' for available commands.")


if __name__ == "__main__":
    main()
