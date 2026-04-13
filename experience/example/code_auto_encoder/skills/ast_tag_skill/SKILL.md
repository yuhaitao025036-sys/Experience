---
name: ast-tag-navigator
description: |
  AST Tag structured code navigation for masked code restoration. Use when working with
  code understanding tasks that require cross-file context: (1) restoring masked/obscured
  code regions, (2) tracing call chains across files, (3) finding symbol definitions
  and references, (4) analyzing lexical parent scopes and child hierarchies.
  Requires ast_tag .jsonl dataset in working directory.
---

# AST Tag Navigator

Structured code navigation using relational AST tags for masked code restoration.

## Core Concept

AST Tag flattens code structure into relational triples:
```
(owner_tag, relation_tag, member_tag)
```

Navigation actions:

| Action | Query Pattern | Use Case |
|--------|---------------|----------|
| `go_to_parent` | `WHERE member_tag = ? AND relation_tag IN LEXICAL` | Find containing scope |
| `expand_children` | `WHERE owner_tag = ?` | List scope contents |
| `find_references` | `WHERE member_tag = ? AND relation_tag IN DYNAMIC` | Cross-file usages |
| `go_to_definition` | Resolve symbol to definition site | Find symbol definition |

## Workflow for Masked Code Restoration

1. **Identify target region**: Note file path and line range of `<Mask>`

2. **Gather nearby symbols**: Query symbols from lines surrounding the mask
   ```sql
   SELECT DISTINCT owner_tag, member_tag
   FROM relations
   WHERE file_id = ? AND line BETWEEN ?-5 AND ?+5
     AND line NOT BETWEEN ? AND ?
   ```

3. **Trace pre-mask definitions**: Find symbols defined before mask that might be used
   ```sql
   SELECT owner_tag, relation_tag, member_tag
   FROM relations
   WHERE file_id = ? AND line < ? AND line > ? - 100
   ORDER BY line DESC LIMIT 30
   ```

4. **Trace post-mask usages**: Find symbols used after mask that depend on it
   ```sql
   SELECT owner_tag, relation_tag, member_tag
   FROM relations
   WHERE file_id = ? AND line > ?
   ORDER BY line ASC LIMIT 30
   ```

5. **Find containing function**: Identify what function/class contains the mask
   ```sql
   SELECT line, owner_tag, member_tag
   FROM relations
   WHERE file_id = ? AND line < ? AND line > ? - 100
     AND (relation_tag LIKE '%FunctionDef%' OR relation_tag LIKE '%ClassDef%')
   ORDER BY line DESC LIMIT 5
   ```

6. **Resolve suspicious symbols**: Use `find_references` on symbols that seem relevant

## SQL Query Helpers

Execute raw SQL against ast_tag database:

```python
db = AstTagDB(conn)
results = db.execute_raw_sql_query("SELECT * FROM relations WHERE ...", params)
```

Common patterns in `references/ast_tag_db_api.md`.

## Relation Tag Classification

**Lexical Relations** (`LEXICAL_RELATION_TAGS`): Static structure
- `Module.body`, `FunctionDef.body`, `If.body`, `While.body`, `For.body`
- `FunctionDef.name`, `FunctionDef.args`, `ClassDef.bases`
- `param`, `import_name`, `except_handler`

**Dynamic Relations** (`DYNAMIC_RELATION_TAGS`): Runtime behavior
- `Call.func`, `Call.args`, `Call.keywords`
- `Assign.targets`, `Assign.value`
- `Attribute.value`, `Attribute.attr`

## Database Loading

Load ast_tag .jsonl files into SQLite:
```python
from experience.ast_tag.ast_tag_db import load_jsonl_dataset_into_ast_tag_db
db = load_jsonl_dataset_into_ast_tag_db("./dataset/")
```

## Script: Convert Python to AST Tags

Pre-process source files to ast_tag format:
```bash
python scripts/convert_to_ast_tag.py <source_dir> <output_dir>
```

## Script: Interactive Navigation

Query the database interactively:
```bash
python scripts/navigate_db.py <dataset_dir>
```

Commands: `refs <symbol>`, `def <symbol>`, `parent <file:tag>`, `children <file:tag>`

For detailed API reference, see `references/ast_tag_db_api.md`.
