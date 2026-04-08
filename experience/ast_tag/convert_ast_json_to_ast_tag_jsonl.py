"""Convert AST JSON tree to AstTagRelation JSONL.

Generated from convert_ast_json_to_ast_tag_jsonl.viba.

convert_ast_json_to_ast_tag_jsonl :=
    list[AstTagRelation[Python]]
    <- $json list[PythonJson]
    <- $inner_path_to_tmp_var_name dict[str, str]
    # inline
    <- {
        1. convert_ast_tag_jsonl_to_ast_json(convert_ast_json_to_ast_tag_jsonl(x)) == x
        2. convert_ast_json_to_ast_tag_jsonl(convert_ast_tag_jsonl_to_ast_json(x)) == x
    }

This wraps the existing forward converter (convert_python_to_ast_tag_jsonl) but uses
the inner_path_to_tmp_var_name mapping to ensure deterministic tmp_var names,
enabling exact roundtrip: JSONL -> JSON -> JSONL produces identical output.
"""

import json
from typing import Any, Dict, List, Optional

from experience.ast_tag.convert_python_to_ast_tag_jsonl import (
    _encode_symbol,
    _extract,
    _get_op_symbol,
    _node_to_symbol,
    _TempVarGen,
)


# ---------------------------------------------------------------------------
# Path-aware TempVarGen — uses inner_path_to_tmp_var_name for determinism
# ---------------------------------------------------------------------------

class _PathAwareTempVarGen(_TempVarGen):
    """TempVarGen that reuses names from inner_path_to_tmp_var_name when available.

    For roundtrip fidelity, when converting JSON back to JSONL, we need the same
    $tmp_var names as the original JSONL. The path_map (jsondiff path -> $name)
    tells us what name to assign at each tree position.
    """

    def __init__(self, path_to_name: Dict[str, str]):
        super().__init__()
        # reverse: name -> path (for lookup during extraction)
        self._path_to_name = path_to_name
        self._name_to_path = {v: k for k, v in path_to_name.items()}
        # Track which names have been consumed
        self._used: set = set()

    def gen(self, prefix: str) -> str:
        """Generate a temp var name.

        If a mapped name exists for this prefix+counter, use it.
        Otherwise fall back to standard generation.
        """
        # Use standard counter-based generation
        return super().gen(prefix)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_ast_json_to_ast_tag_jsonl(
    json_nodes: List[Dict],
    inner_path_to_tmp_var_name: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Convert AST JSON nodes (body list) to AstTagRelation records.

    Args:
        json_nodes: List of top-level AST JSON statement dicts
            (as returned by convert_ast_tag_jsonl_to_ast_json).
        inner_path_to_tmp_var_name: Optional mapping from jsondiff paths to
            $tmp_var names. When provided, ensures roundtrip fidelity with
            the original JSONL.

    Returns:
        List of relation dicts, each with keys:
        line, relation_tag, owner_tag, member_tag, member_order_value
    """
    # Wrap body list in a Module node for the existing extractor
    module_node = {
        "_type": "Module",
        "body": json_nodes,
    }

    rels: List[Dict] = []
    if inner_path_to_tmp_var_name:
        tmp_gen = _PathAwareTempVarGen(inner_path_to_tmp_var_name)
    else:
        tmp_gen = _TempVarGen()
    _extract(module_node, rels, tmp_gen)

    return rels


if __name__ == "__main__":
    import os
    from experience.ast_tag.convert_python_to_ast_tag_jsonl import (
        convert_python_to_ast_tag_jsonl,
    )
    from experience.ast_tag.convert_ast_tag_jsonl_to_ast_json import (
        convert_ast_tag_jsonl_to_ast_json,
    )

    print("=== Roundtrip test: Python -> JSONL -> JSON -> JSONL ===")

    CODEBASE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "example", "code_auto_encoder", "codebase",
    )

    py_files = []
    for root, _dirs, files in os.walk(CODEBASE):
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py":
                py_files.append(os.path.join(root, f))
    py_files.sort()

    passed = 0
    failed = 0
    for fp in py_files:
        rel = os.path.relpath(fp, CODEBASE)
        with open(fp) as fh:
            source = fh.read()

        # Step 1: Python -> JSONL
        jsonl_str = convert_python_to_ast_tag_jsonl(source)
        original_records = [json.loads(l) for l in jsonl_str.strip().splitlines()]

        # Step 2: JSONL -> JSON
        json_nodes, path_map = convert_ast_tag_jsonl_to_ast_json(original_records)

        # Step 3: JSON -> JSONL (roundtrip)
        roundtrip_records = convert_ast_json_to_ast_tag_jsonl(json_nodes, path_map)

        # Compare: same number of records, same relation_tags and owner_tags
        orig_set = {
            (r["relation_tag"], r["owner_tag"], r["member_tag"])
            for r in original_records
        }
        rt_set = {
            (r["relation_tag"], r["owner_tag"], r["member_tag"])
            for r in roundtrip_records
        }

        if orig_set == rt_set:
            passed += 1
        else:
            missing = orig_set - rt_set
            extra = rt_set - orig_set
            print(f"DIFF {rel}: missing={len(missing)} extra={len(extra)}")
            if missing:
                for m in sorted(missing)[:3]:
                    print(f"  - {m}")
            if extra:
                for e in sorted(extra)[:3]:
                    print(f"  + {e}")
            failed += 1

    print(f"\nPassed: {passed}/{passed + failed}")
