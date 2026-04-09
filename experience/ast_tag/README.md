# AST Tag: Structured Code Representation for Agentic RL

A Python library for converting source code into a relational tag format, enabling structured code navigation and manipulation through database queries.

## Motivation

### The Code Navigation Problem

When humans read code, they constantly navigate between:
- **Lexical scope**: "What function/class contains this line?" (compile-time structure)
- **Dynamic scope**: "Where is this function called?" (runtime behavior)

This navigation follows predictable patterns that can be modeled as action sequences. We formalize these patterns as an **action table**:

| | Forward | Reverse |
|---|---|---|
| **Lexical Scope** | Expand Children | Go to Parent |
| **Dynamic Scope** | Go to Definition | Find All References |

### Agentic RL for Code Understanding

The core insight: **code navigation traces are trajectories** that can be optimized.

```
[LLM query -> DB query]+ -> LLM query -> loss/reward vs ground truth
```

1. An agent starts with a masked code region
2. Issues navigation queries (lexical/dynamic, forward/reverse)
3. Accumulates context from query results
4. Predicts the masked content
5. Receives reward based on accuracy

This enables:
- **Training**: Optimize query strategies via reinforcement learning
- **Evaluation**: Measure how efficiently an agent finds relevant context
- **Interpretability**: Query traces reveal reasoning patterns

## Architecture

### AstTagRecord Schema

Each record captures a parent-child relationship in the AST:

```python
{
    "file_id": "main.py.jsonl",       # Source file identifier
    "line": 10,
    "relation_tag": "Call.func",      # Type.field_name format
    "owner_tag": "$Call_3",           # Unique node identifier
    "member_tag": "make_tensor",      # Child value (symbol or temp var)
    "member_order_value": 0           # Position in ordered fields
}
```

### Relation Tag Classification

Tags are classified into two categories:

- **Lexical Relations** (`LEXICAL_RELATION_TAGS`): Static containment structure
  - Block bodies: `Module.body`, `FunctionDef.body`, `If.body`, ...
  - Definition structure: `FunctionDef.name`, `ClassDef.bases`, ...
  - Parameters, imports, exception handlers

- **Dynamic Relations** (`DYNAMIC_RELATION_TAGS`): Runtime execution flow
  - Calls: `Call.func`, `Call.args`, ...
  - Assignments: `Assign.targets`, `Assign.value`, ...
  - Operators, control flow expressions, attribute access

### Tag Actions

Four fundamental navigation primitives:

| Action | Direction | Scope | Query Pattern |
|--------|-----------|-------|---------------|
| `lexical_scope_expand_children` | Forward | Lexical | `SELECT WHERE owner_tag = ?` |
| `lexical_scope_go_to_parent` | Reverse | Lexical | `SELECT WHERE member_tag = ?` |
| `dynamic_scope_go_to_definition` | Forward | Dynamic | Resolve symbol to definition site |
| `dynamic_scope_find_all_references` | Reverse | Dynamic | `SELECT WHERE member_tag = ?` across files |

## Usage

### Convert Python to AST Tags

```python
from experience.ast_tag.convert_python_to_ast_tag_jsonl import convert_python_to_ast_tag_jsonl

source = '''
def greet(name: str) -> str:
    return f"Hello, {name}!"
'''

jsonl = convert_python_to_ast_tag_jsonl(source)
# Each line is a JSON AstTagRecord
```

### Load into Database

```python
from experience.ast_tag.ast_tag_db import load_jsonl_dataset_into_ast_tag_db

db = load_jsonl_dataset_into_ast_tag_db("./dataset/")
# Returns AstTagDB ready for queries
```

### Navigate Code

```python
from experience.ast_tag.tag_actions import (
    lexical_scope_go_to_parent,
    dynamic_scope_find_all_references,
)

# Find the function containing a call
parent = lexical_scope_go_to_parent(db, "main.jsonl", "$Call_5")

# Find all callers of a function
refs = dynamic_scope_find_all_references(db, "make_tensor")
```

## Files

```
experience/ast_tag/
├── ast_tag_record.viba        # Schema definition
├── ast_tag_db.py              # SQLite database interface
├── convert_python_to_ast_tag_jsonl.py    # Python -> tags
├── convert_ast_tag_jsonl_to_python.py    # Tags -> Python
├── relation_tag_classification.viba      # Lexical vs Dynamic tags
├── tag_actions/               # Navigation primitives
│   ├── lexical_scope_expand_children.py
│   ├── lexical_scope_go_to_parent.py
│   ├── dynamic_scope_go_to_definition.py
│   └── dynamic_scope_find_all_references.py
└── test_dataset/              # Example JSONL files
```

## Design Principles

1. **Flat Relations**: AST tree is flattened into (owner, member) pairs
2. **Cross-file Symbols**: Symbols like function names bridge files without line numbers
3. **Type-aligned Tags**: `Type.field_name` format is self-documenting
4. **Database-native**: Designed for SQL queries, enabling efficient indexing

## Future Work

- [ ] Implement complete `dynamic_scope_go_to_definition` with cross-file resolution
- [ ] Add re-export tracing to `dynamic_scope_find_all_references`
- [ ] Build RL environment for navigation policy optimization
- [ ] Support multiple programming languages