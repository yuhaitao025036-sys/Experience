"""Convert AST JSON tree to AstTagRelation JSONL.

Thin wrapper: wraps body list in a Module node and runs the generic extractor.
"""

import json
from typing import Any, Dict, List, Optional

from experience.ast_tag.convert_python_to_ast_tag_jsonl import (
    _TempVarGen, _extract,
)


def convert_ast_json_to_ast_tag_jsonl(
    json_nodes: List[Dict],
    inner_path_to_tmp_var_name: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    module_node = {"_type": "Module", "body": json_nodes}
    rels: List[Dict] = []
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
    for fp in py_files:
        rel = os.path.relpath(fp, CODEBASE)
        with open(fp) as fh:
            source = fh.read()
        jsonl_str = convert_python_to_ast_tag_jsonl(source)
        original = [json.loads(l) for l in jsonl_str.strip().splitlines()]

        json_nodes, path_map = convert_ast_tag_jsonl_to_ast_json(original)
        roundtrip = convert_ast_json_to_ast_tag_jsonl(json_nodes, path_map)

        orig_set = {(r["relation_tag"], r["owner_tag"], r["member_tag"])
                     for r in original}
        rt_set = {(r["relation_tag"], r["owner_tag"], r["member_tag"])
                   for r in roundtrip}
        if orig_set == rt_set:
            passed += 1
        else:
            print(f"DIFF {rel}: -{len(orig_set - rt_set)} +{len(rt_set - orig_set)}")
    print(f"\nRoundtrip: {passed}/{len(py_files)}")
