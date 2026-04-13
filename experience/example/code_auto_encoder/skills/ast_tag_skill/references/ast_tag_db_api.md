# AST Tag DB API Reference

## AstTagDB Class

Wrapper around `sqlite3.Connection` for AST tag relation queries.

### Methods

#### `get_all_loaded_file_ids() -> list[str]`
Returns all loaded JSONL file identifiers, sorted alphabetically.

```python
file_ids = db.get_all_loaded_file_ids()
```

#### `count_file_relation_records(file_id: str) -> int`
Total relation rows in a file (proxy for file complexity).

```python
count = db.count_file_relation_records("module.py.jsonl")
```

#### `get_nearby_symbols_around_line_range(file_id, line_start, line_end, context_margin=5) -> list[str]`
Collect symbols from lines surrounding a line range (excluding the range itself).

```python
symbols = db.get_nearby_symbols_around_line_range(
    "module.py.jsonl",
    line_range_start=10,
    line_range_end=15,
    context_margin=5
)
```

#### `execute_raw_sql_query(sql_statement: str, query_params: tuple = ()) -> list[tuple]`
Raw SQL escape hatch for ad-hoc analysis.

```python
results = db.execute_raw_sql_query(
    "SELECT * FROM relations WHERE line > ? LIMIT 10",
    (50,)
)
```

## Navigation Functions

### Lexical Scope Navigation

#### `lexical_scope_go_to_parent(ast_tag_db, file_id, member_tag) -> CodePosition`
Find the lexical parent scope containing a symbol. Returns exactly one CodePosition.

```python
from experience.ast_tag.tag_actions.lexical_scope_go_to_parent import lexical_scope_go_to_parent

parent = lexical_scope_go_to_parent(db, "module.py.jsonl", "$Call_5")
```

#### `lexical_scope_expand_children(ast_tag_db, file_id, owner_tag) -> list[CodePosition]`
Return all child nodes of a scope.

```python
from experience.ast_tag.tag_actions.lexical_scope_expand_children import lexical_scope_expand_children

children = lexical_scope_expand_children(db, "module.py.jsonl", "$FunctionDef_0")
```

### Dynamic Scope Navigation

#### `dynamic_scope_find_all_references(ast_tag_db, symbol_name) -> list[CodePosition]`
Search ALL files for references to a symbol.

```python
from experience.ast_tag.tag_actions.dynamic_scope_find_all_references import dynamic_scope_find_all_references

refs = dynamic_scope_find_all_references(db, "make_tensor")
```

#### `dynamic_scope_go_to_definition(ast_tag_db, file_id, member_tag) -> Optional[CodePosition]`
Resolve a symbol to its definition site.

```python
from experience.ast_tag.tag_actions.dynamic_scope_go_to_definition import dynamic_scope_go_to_definition

defn = dynamic_scope_go_to_definition(db, "module.py.jsonl", "make_tensor")
```

## CodePosition NamedTuple

```python
from experience.ast_tag.tag_actions.code_position import CodePosition

cp = CodePosition(
    file_id="module.py.jsonl",
    line=10,
    owner_tag="$Call_3",
    relation_tag="Call.func",
    member_tag="make_tensor",
    member_order_value=0
)
```

Fields:
- `file_id`: File identifier
- `line`: Line number
- `owner_tag`: Parent node tag
- `relation_tag`: Relationship type
- `member_tag`: Child value
- `member_order_value`: Position in ordered fields

## Loading Database

```python
from experience.ast_tag.ast_tag_db import load_jsonl_dataset_into_ast_tag_db

db = load_jsonl_dataset_into_ast_tag_db(
    dataset_dir="./codebase_ast_tags",
    db_path=":memory:"  # or a file path like "./ast_tags.db"
)
```

## Relation Tag Constants

```python
from experience.ast_tag.relation_tag_classification import (
    LEXICAL_RELATION_TAGS,
    DYNAMIC_RELATION_TAGS
)

# Lexical: static containment structure
print(LEXICAL_RELATION_TAGS[:10])
# ['Module.body', 'FunctionDef.body', 'AsyncFunctionDef.body', ...]

# Dynamic: runtime execution flow
print(DYNAMIC_RELATION_TAGS[:10])
# ['Call.func', 'Call.args', 'Call.keywords', ...]
```

## Common SQL Queries

### Find all function calls in a file
```sql
SELECT owner_tag, member_tag FROM relations
WHERE file_id = ? AND relation_tag = 'Call.func'
```

### Find all assignments to a variable
```sql
SELECT line, value FROM relations
WHERE file_id = ? AND member_tag = 'variable_name'
  AND relation_tag = 'Assign.targets'
```

### Find containing class for a function
```sql
SELECT line, member_tag FROM relations
WHERE file_id = ? AND line < ?
  AND relation_tag = 'ClassDef.name'
ORDER BY line DESC LIMIT 1
```

### Count relations by type
```sql
SELECT relation_tag, COUNT(*) FROM relations
GROUP BY relation_tag
ORDER BY COUNT(*) DESC
```
