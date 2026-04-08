"""Compute normalized edit distance between two JSON trees.

Uses jsonpatch to diff, counts leaf values touched by patch ops,
normalizes by total leaf count of both trees.
"""

from typing import Any, Dict, List

import jsonpatch


def _num_leaves(obj: Any) -> int:
    """Count leaf (non-container) values in a JSON-compatible object.

    Empty containers count as 1 leaf (they carry structural information).
    """
    if isinstance(obj, dict):
        return sum(_num_leaves(v) for v in obj.values()) if obj else 1
    if isinstance(obj, list):
        return sum(_num_leaves(v) for v in obj) if obj else 1
    return 1


def _resolve_pointer(doc: Any, path: str) -> Any:
    """Resolve a JSON pointer path against a document (RFC 6901)."""
    if not path or path == "/":
        return doc
    parts = path.lstrip("/").split("/")
    cur = doc
    for p in parts:
        # RFC 6901 unescaping: ~1 → /, ~0 → ~ (order matters)
        p = p.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(p)]
        elif isinstance(cur, dict):
            cur = cur[p]
        else:
            return None
    return cur


def _patch_num_leaves(patch: jsonpatch.JsonPatch, lhs: Any) -> int:
    """Sum leaf counts across all patch ops (add/remove/replace values).

    For remove ops, we must simulate sequential application because
    jsonpatch paths assume prior ops have already been applied.
    E.g. two 'remove /0' on [A, B, C] removes A then B, not A twice.

    For replace ops, we count both old and new value leaves, consistent
    with the equivalent remove+add decomposition.
    """
    import json
    doc = json.loads(json.dumps(lhs))  # faster than copy.deepcopy for JSON data
    total = 0
    for op in patch.patch:
        kind = op.get("op", "")
        if kind == "remove":
            removed = _resolve_pointer(doc, op.get("path", ""))
            total += _num_leaves(removed)
            # Apply the remove so subsequent paths resolve correctly
            try:
                jsonpatch.JsonPatch([op]).apply(doc, in_place=True)
            except Exception:
                pass
        elif kind == "add":
            total += _num_leaves(op.get("value"))
            try:
                jsonpatch.JsonPatch([op]).apply(doc, in_place=True)
            except Exception:
                pass
        elif kind == "replace":
            old_val = _resolve_pointer(doc, op.get("path", ""))
            total += _num_leaves(old_val) + _num_leaves(op.get("value"))
            try:
                jsonpatch.JsonPatch([op]).apply(doc, in_place=True)
            except Exception:
                pass
        elif kind == "move" or kind == "copy":
            total += 1
            try:
                jsonpatch.JsonPatch([op]).apply(doc, in_place=True)
            except Exception:
                pass
    return total


def _json_safe(obj: Any) -> Any:
    """Convert non-JSON-serializable values (Ellipsis, bytes, etc.) for jsonpatch."""
    if obj is ...:
        return "..."
    if isinstance(obj, bytes):
        return repr(obj)
    if isinstance(obj, complex):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(_json_safe(x) for x in obj)
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def get_json_distance(lhs: Any, rhs: Any) -> float:
    """Normalized JSON distance: 0.0 = identical, approaches 1.0 = fully different.

    Averages both directions to guarantee symmetry:
    d(a,b) == d(b,a).
    """
    lhs = _json_safe(lhs)
    rhs = _json_safe(rhs)
    total = _num_leaves(lhs) + _num_leaves(rhs)
    if total == 0:
        return 0.0
    p_fwd = _patch_num_leaves(jsonpatch.make_patch(lhs, rhs), lhs)
    p_rev = _patch_num_leaves(jsonpatch.make_patch(rhs, lhs), rhs)
    return (p_fwd + p_rev) / (2 * total)


if __name__ == "__main__":
    # Test 1: identical
    a = {"_type": "FunctionDef", "name": "foo", "body": [{"_type": "Return", "value": 1}]}
    assert get_json_distance(a, a) == 0.0, "identical should be 0.0"
    print(f"Test 1 (identical):  {get_json_distance(a, a)}")

    # Test 2: completely different
    b = {"_type": "ClassDef", "name": "Bar", "bases": ["object"]}
    d = get_json_distance(a, b)
    print(f"Test 2 (different):  {d:.4f}")
    assert d > 0.0, "different should be > 0.0"

    # Test 3: partially different
    c = {"_type": "FunctionDef", "name": "bar", "body": [{"_type": "Return", "value": 2}]}
    d3 = get_json_distance(a, c)
    print(f"Test 3 (partial):    {d3:.4f}")
    assert 0.0 < d3 < 1.0, f"partial diff should be between 0 and 1, got {d3}"

    # Test 4: empty vs empty
    assert get_json_distance({}, {}) == 0.0, "empty vs empty should be 0.0"
    print(f"Test 4 (empty):      {get_json_distance({}, {})}")

    # Test 5: empty vs non-empty
    d5 = get_json_distance({}, {"a": 1})
    print(f"Test 5 (empty vs 1): {d5:.4f}")
    assert d5 > 0.0

    # Test 6: list ordering matters
    x = [1, 2, 3]
    y = [1, 3, 2]
    d6 = get_json_distance(x, y)
    print(f"Test 6  (list order):      {d6:.4f}")
    assert d6 > 0.0

    # Test 7: symmetry — distance(a,b) == distance(b,a)
    d7a = get_json_distance(a, b)
    d7b = get_json_distance(b, a)
    print(f"Test 7  (symmetry):        {d7a:.4f} vs {d7b:.4f}")
    assert abs(d7a - d7b) < 1e-9, "distance should be symmetric"

    # Test 8: nested identical
    deep = {"a": {"b": {"c": [1, 2, {"d": 3}]}}}
    assert get_json_distance(deep, deep) == 0.0
    print(f"Test 8  (nested ident):    {get_json_distance(deep, deep)}")

    # Test 9: single leaf change in deep tree
    deep2 = {"a": {"b": {"c": [1, 2, {"d": 99}]}}}
    d9 = get_json_distance(deep, deep2)
    print(f"Test 9  (deep 1 leaf):     {d9:.4f}")
    assert 0.0 < d9 < 0.5, "one leaf change in 6-leaf tree should be small"

    # Test 10: list append
    d10 = get_json_distance([1, 2], [1, 2, 3])
    print(f"Test 10 (list append):     {d10:.4f}")
    assert 0.0 < d10 < 1.0

    # Test 11: list delete
    d11 = get_json_distance([1, 2, 3], [1, 3])
    print(f"Test 11 (list delete):     {d11:.4f}")
    assert 0.0 < d11 < 1.0

    # Test 12: real AST — add a decorator
    fn1 = {"_type": "FunctionDef", "name": "f", "args": {"_type": "arguments", "args": []},
           "body": [{"_type": "Pass"}], "decorator_list": []}
    fn2 = {"_type": "FunctionDef", "name": "f", "args": {"_type": "arguments", "args": []},
           "body": [{"_type": "Pass"}], "decorator_list": ["staticmethod"]}
    d12 = get_json_distance(fn1, fn2)
    print(f"Test 12 (add decorator):   {d12:.4f}")
    assert 0.0 < d12 < 0.5

    # Test 13: real AST — rename function
    fn3 = {"_type": "FunctionDef", "name": "g", "args": {"_type": "arguments", "args": []},
           "body": [{"_type": "Pass"}], "decorator_list": []}
    d13 = get_json_distance(fn1, fn3)
    print(f"Test 13 (rename func):     {d13:.4f}")
    assert 0.0 < d13 < 0.5

    # Test 14: real AST — change body statement
    fn4 = {"_type": "FunctionDef", "name": "f", "args": {"_type": "arguments", "args": []},
           "body": [{"_type": "Return", "value": 0}], "decorator_list": []}
    d14 = get_json_distance(fn1, fn4)
    print(f"Test 14 (change body):     {d14:.4f}")
    assert 0.0 < d14 < 1.0

    # Test 15: scalar vs scalar
    d15 = get_json_distance(42, 42)
    assert d15 == 0.0
    print(f"Test 15 (scalar same):     {d15}")

    # Test 16: scalar vs different scalar
    d16 = get_json_distance(42, 99)
    print(f"Test 16 (scalar diff):     {d16:.4f}")
    assert d16 > 0.0

    # Test 17: list vs empty list
    d17 = get_json_distance([1, 2, 3], [])
    print(f"Test 17 (list vs empty):   {d17:.4f}")
    assert d17 > 0.0

    # Test 18: triangle inequality spot check — d(a,c) <= d(a,b) + d(b,c)
    d_ac = get_json_distance(a, c)
    d_ab = get_json_distance(a, b)
    d_bc = get_json_distance(b, c)
    print(f"Test 18 (triangle ineq):   d(a,c)={d_ac:.4f} <= d(a,b)+d(b,c)={d_ab+d_bc:.4f}")
    assert d_ac <= d_ab + d_bc + 1e-9, "triangle inequality violated"

    # Test 19: real roundtrip — convert a .jsonl file, compare original vs reconstructed
    import json, os
    from experience.ast_tag.convert_python_to_ast_tag_jsonl import convert_python_to_ast_tag_jsonl, ast_to_dict
    from experience.ast_tag.convert_ast_tag_jsonl_to_ast_json import convert_ast_tag_jsonl_to_ast_json
    import ast as _ast
    codebase = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "example", "code_auto_encoder", "codebase")
    sample = os.path.join(codebase, "fs_util", "pack_dir.py")
    with open(sample) as fh:
        source = fh.read()
    tree = _ast.parse(source)
    original_json = ast_to_dict(tree)["body"]
    jsonl_str = convert_python_to_ast_tag_jsonl(source)
    records = [json.loads(l) for l in jsonl_str.strip().splitlines()]
    recon_json, _ = convert_ast_tag_jsonl_to_ast_json(records)
    d19 = get_json_distance(original_json, recon_json)
    print(f"Test 19 (roundtrip dist):  {d19:.4f}")
    assert d19 < 0.6, f"roundtrip should be reasonably close, got {d19}"

    # Test 20: large identical lists
    big = list(range(100))
    assert get_json_distance(big, big) == 0.0
    print(f"Test 20 (big list ident):  {get_json_distance(big, big)}")

    # Test 21: one element changed in big list
    big2 = list(range(100)); big2[50] = 999
    d21 = get_json_distance(big, big2)
    print(f"Test 21 (big 1 change):    {d21:.4f}")
    assert 0.0 < d21 < 0.1

    # Test 22: string values — replace counts both old+new, so 1 leaf replaced = distance 1.0
    d22 = get_json_distance({"s": "hello"}, {"s": "world"})
    print(f"Test 22 (string diff):     {d22:.4f}")
    assert d22 == 1.0

    # Test 23: string identical
    assert get_json_distance("abc", "abc") == 0.0
    print(f"Test 23 (string same):     0.0")

    # Test 24: null vs value
    d24 = get_json_distance({"a": None}, {"a": 1})
    print(f"Test 24 (null vs val):     {d24:.4f}")
    assert d24 > 0.0

    # Test 25: bool values — replace counts both old+new
    d25 = get_json_distance(True, False)
    print(f"Test 25 (bool diff):       {d25:.4f}")
    assert d25 == 1.0

    # Test 26: nested list append
    d26 = get_json_distance({"a": [1, 2]}, {"a": [1, 2, 3]})
    print(f"Test 26 (nested append):   {d26:.4f}")
    assert 0.0 < d26 < 0.5

    # Test 27: key added to dict
    d27 = get_json_distance({"a": 1}, {"a": 1, "b": 2})
    print(f"Test 27 (key added):       {d27:.4f}")
    assert 0.0 < d27 < 1.0

    # Test 28: key removed from dict
    d28 = get_json_distance({"a": 1, "b": 2}, {"a": 1})
    print(f"Test 28 (key removed):     {d28:.4f}")
    assert d28 == d27, "add/remove should be symmetric"

    # Test 29: deeply nested single change — replace counts both old+new = 1.0
    d29_a = {"l1": {"l2": {"l3": {"l4": {"l5": "deep"}}}}}
    d29_b = {"l1": {"l2": {"l3": {"l4": {"l5": "changed"}}}}}
    d29 = get_json_distance(d29_a, d29_b)
    print(f"Test 29 (deep nest chg):   {d29:.4f}")
    assert d29 == 1.0, "single leaf tree, replacing the only leaf = fully different"

    # Test 30: mixed types in list
    mix1 = [1, "two", True, None, {"k": "v"}]
    mix2 = [1, "two", True, None, {"k": "v"}]
    assert get_json_distance(mix1, mix2) == 0.0
    print(f"Test 30 (mixed ident):     0.0")

    # Test 31: mixed types with diff
    mix3 = [1, "TWO", False, None, {"k": "x"}]
    d31 = get_json_distance(mix1, mix3)
    print(f"Test 31 (mixed diff):      {d31:.4f}")
    assert 0.0 < d31 < 1.0

    # Test 32: real AST — If node
    if1 = {"_type": "If", "test": {"_type": "Name", "id": "x"},
           "body": [{"_type": "Pass"}], "orelse": []}
    if2 = {"_type": "If", "test": {"_type": "Name", "id": "y"},
           "body": [{"_type": "Pass"}], "orelse": []}
    d32 = get_json_distance(if1, if2)
    print(f"Test 32 (If rename var):   {d32:.4f}")
    assert 0.0 < d32 < 0.3

    # Test 33: real AST — If add else branch
    if3 = {"_type": "If", "test": {"_type": "Name", "id": "x"},
           "body": [{"_type": "Pass"}],
           "orelse": [{"_type": "Return", "value": 0}]}
    d33 = get_json_distance(if1, if3)
    print(f"Test 33 (If add else):     {d33:.4f}")
    assert 0.0 < d33 < 0.5

    # Test 34: real AST — Call with args
    call1 = {"_type": "Call", "func": "print", "args": ["hello"], "keywords": []}
    call2 = {"_type": "Call", "func": "print", "args": ["hello", "world"], "keywords": []}
    d34 = get_json_distance(call1, call2)
    print(f"Test 34 (Call add arg):    {d34:.4f}")
    assert 0.0 < d34 < 0.5

    # Test 35: real AST — Call change func name
    call3 = {"_type": "Call", "func": "log", "args": ["hello"], "keywords": []}
    d35 = get_json_distance(call1, call3)
    print(f"Test 35 (Call chg func):   {d35:.4f}")
    assert 0.0 < d35 < 0.5

    # Test 36: real AST — Assign
    asgn1 = {"_type": "Assign", "targets": [{"_type": "Name", "id": "x"}],
             "value": {"_type": "Constant", "value": 1}}
    asgn2 = {"_type": "Assign", "targets": [{"_type": "Name", "id": "x"}],
             "value": {"_type": "Constant", "value": 2}}
    d36 = get_json_distance(asgn1, asgn2)
    print(f"Test 36 (Assign chg val):  {d36:.4f}")
    assert 0.0 < d36 < 0.3

    # Test 37: real AST — Import vs ImportFrom
    imp1 = {"_type": "Import", "names": [{"_type": "alias", "name": "os"}]}
    imp2 = {"_type": "ImportFrom", "module": "os", "names": [{"_type": "alias", "name": "path"}]}
    d37 = get_json_distance(imp1, imp2)
    print(f"Test 37 (Import types):    {d37:.4f}")
    assert d37 > 0.0

    # Test 38: monotonicity — more changes = larger distance
    base = {"a": 1, "b": 2, "c": 3, "d": 4}
    chg1 = {"a": 1, "b": 2, "c": 3, "d": 99}
    chg2 = {"a": 1, "b": 2, "c": 99, "d": 99}
    chg3 = {"a": 1, "b": 99, "c": 99, "d": 99}
    d38a = get_json_distance(base, chg1)
    d38b = get_json_distance(base, chg2)
    d38c = get_json_distance(base, chg3)
    print(f"Test 38 (monotone):        {d38a:.4f} < {d38b:.4f} < {d38c:.4f}")
    assert d38a < d38b < d38c, "more changes should yield larger distance"

    # Test 39: list reversal
    d39 = get_json_distance([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    print(f"Test 39 (list reverse):    {d39:.4f}")
    assert d39 > 0.0

    # Test 40: real AST — BinOp change operator
    binop1 = {"_type": "BinOp", "left": {"_type": "Name", "id": "a"},
              "op": {"_type": "Add"}, "right": {"_type": "Name", "id": "b"}}
    binop2 = {"_type": "BinOp", "left": {"_type": "Name", "id": "a"},
              "op": {"_type": "Mult"}, "right": {"_type": "Name", "id": "b"}}
    d40 = get_json_distance(binop1, binop2)
    print(f"Test 40 (BinOp chg op):    {d40:.4f}")
    assert 0.0 < d40 < 0.3

    # Test 41: real AST — ClassDef add base
    cls1 = {"_type": "ClassDef", "name": "Foo", "bases": [],
            "body": [{"_type": "Pass"}], "keywords": [], "decorator_list": []}
    cls2 = {"_type": "ClassDef", "name": "Foo", "bases": ["object"],
            "body": [{"_type": "Pass"}], "keywords": [], "decorator_list": []}
    d41 = get_json_distance(cls1, cls2)
    print(f"Test 41 (Class add base):  {d41:.4f}")
    assert 0.0 < d41 < 0.3

    # Test 42: real AST — full module body diff
    mod1 = [imp1, asgn1]
    mod2 = [imp2, asgn2]
    d42 = get_json_distance(mod1, mod2)
    print(f"Test 42 (module body):     {d42:.4f}")
    assert 0.0 < d42 < 1.0

    # Test 43: distance(x, x copy) == 0
    import copy
    deep_copy = copy.deepcopy(binop1)
    assert get_json_distance(binop1, deep_copy) == 0.0
    print(f"Test 43 (deepcopy ident):  0.0")

    # Test 44: real roundtrip on a different file
    sample2 = os.path.join(codebase, "sparse_util", "group_random_select.py")
    with open(sample2) as fh:
        source2 = fh.read()
    tree2 = _ast.parse(source2)
    orig2 = ast_to_dict(tree2)["body"]
    jsonl2 = convert_python_to_ast_tag_jsonl(source2)
    recs2 = [json.loads(l) for l in jsonl2.strip().splitlines()]
    recon2, _ = convert_ast_tag_jsonl_to_ast_json(recs2)
    d44 = get_json_distance(orig2, recon2)
    print(f"Test 44 (roundtrip #2):    {d44:.4f}")
    assert d44 < 0.6

    # Test 45: real roundtrip on a large file
    sample3 = os.path.join(codebase, "symbolic_tensor", "function", "st_moe_backward.py")
    with open(sample3) as fh:
        source3 = fh.read()
    tree3 = _ast.parse(source3)
    orig3 = ast_to_dict(tree3)["body"]
    jsonl3 = convert_python_to_ast_tag_jsonl(source3)
    recs3 = [json.loads(l) for l in jsonl3.strip().splitlines()]
    recon3, _ = convert_ast_tag_jsonl_to_ast_json(recs3)
    d45 = get_json_distance(orig3, recon3)
    print(f"Test 45 (roundtrip big):   {d45:.4f}")
    assert d45 < 0.6

    # Test 46: empty dict vs nested dict — empty container = 1 leaf now
    d46 = get_json_distance({}, {"a": {"b": {"c": 1}}})
    print(f"Test 46 (empty vs nest):   {d46:.4f}")
    assert d46 == 0.5

    # Test 47: identical nested lists of dicts
    lod = [{"x": i, "y": i*2} for i in range(10)]
    assert get_json_distance(lod, lod) == 0.0
    print(f"Test 47 (list-of-dict):    0.0")

    # Test 48: one dict changed in list of dicts
    lod2 = [{"x": i, "y": i*2} for i in range(10)]
    lod2[5] = {"x": 5, "y": 999}
    d48 = get_json_distance(lod, lod2)
    print(f"Test 48 (lod 1 change):    {d48:.4f}")
    assert 0.0 < d48 < 0.1

    # Test 49: symmetry on real AST nodes
    d49a = get_json_distance(fn1, fn4)
    d49b = get_json_distance(fn4, fn1)
    print(f"Test 49 (AST symmetry):    {d49a:.4f} vs {d49b:.4f}")
    assert abs(d49a - d49b) < 1e-9

    # Test 50: distance between two different real files
    d50 = get_json_distance(orig2, orig3)
    print(f"Test 50 (cross-file):      {d50:.4f}")
    assert d50 > 0.3, "different files should have significant distance"

    # Test 51: self-distance is always 0 on real AST
    assert get_json_distance(orig2, orig2) == 0.0
    print(f"Test 51 (real self=0):     0.0")

    # Test 52: list prefix — superset contains subset
    d52 = get_json_distance([1, 2, 3], [1, 2, 3, 4, 5])
    print(f"Test 52 (list prefix):     {d52:.4f}")
    assert 0.0 < d52 < 0.5

    # Test 53: list suffix removal
    d53 = get_json_distance([1, 2, 3, 4, 5], [1, 2, 3])
    print(f"Test 53 (list suffix rm):  {d53:.4f}")
    assert abs(d52 - d53) < 1e-9, "prefix add vs suffix remove should be symmetric"

    # Test 54: dict key rename (remove + add)
    d54 = get_json_distance({"old_key": 1}, {"new_key": 1})
    print(f"Test 54 (key rename):      {d54:.4f}")
    assert 0.0 < d54 <= 1.0

    # Test 55: nested dict key rename
    d55 = get_json_distance({"a": {"old": 1}}, {"a": {"new": 1}})
    print(f"Test 55 (nest key rename): {d55:.4f}")
    assert 0.0 < d55 < 1.0

    # Test 56: type change int→str same content
    d56 = get_json_distance({"v": 42}, {"v": "42"})
    print(f"Test 56 (int vs str):      {d56:.4f}")
    assert d56 > 0.0

    # Test 57: None vs 0
    d57 = get_json_distance(None, 0)
    print(f"Test 57 (None vs 0):       {d57:.4f}")
    assert d57 > 0.0

    # Test 58: None vs None
    assert get_json_distance(None, None) == 0.0
    print(f"Test 58 (None vs None):    0.0")

    # Test 59: real AST — For loop
    for1 = {"_type": "For", "target": {"_type": "Name", "id": "i"},
            "iter": {"_type": "Call", "func": "range", "args": [10]},
            "body": [{"_type": "Pass"}], "orelse": []}
    for2 = {"_type": "For", "target": {"_type": "Name", "id": "j"},
            "iter": {"_type": "Call", "func": "range", "args": [20]},
            "body": [{"_type": "Pass"}], "orelse": []}
    d59 = get_json_distance(for1, for2)
    print(f"Test 59 (For loop diff):   {d59:.4f}")
    assert 0.0 < d59 < 0.3

    # Test 60: real AST — While vs For
    while1 = {"_type": "While", "test": {"_type": "Constant", "value": True},
              "body": [{"_type": "Pass"}], "orelse": []}
    d60 = get_json_distance(for1, while1)
    print(f"Test 60 (For vs While):    {d60:.4f}")
    assert d60 > d59, "different node types should be farther"

    # Test 61: real AST — Try/Except
    try1 = {"_type": "Try", "body": [{"_type": "Pass"}],
            "handlers": [{"_type": "ExceptHandler", "type": {"_type": "Name", "id": "Exception"},
                          "name": "e", "body": [{"_type": "Pass"}]}],
            "orelse": [], "finalbody": []}
    try2 = {"_type": "Try", "body": [{"_type": "Pass"}],
            "handlers": [{"_type": "ExceptHandler", "type": {"_type": "Name", "id": "ValueError"},
                          "name": "e", "body": [{"_type": "Pass"}]}],
            "orelse": [], "finalbody": []}
    d61 = get_json_distance(try1, try2)
    print(f"Test 61 (Try exc type):    {d61:.4f}")
    assert 0.0 < d61 < 0.2

    # Test 62: real AST — add finally block
    try3 = {"_type": "Try", "body": [{"_type": "Pass"}],
            "handlers": [{"_type": "ExceptHandler", "type": {"_type": "Name", "id": "Exception"},
                          "name": "e", "body": [{"_type": "Pass"}]}],
            "orelse": [], "finalbody": [{"_type": "Pass"}]}
    d62 = get_json_distance(try1, try3)
    print(f"Test 62 (Try add finally): {d62:.4f}")
    assert 0.0 < d62 < 0.3

    # Test 63: real AST — List comprehension
    lc1 = {"_type": "ListComp", "elt": {"_type": "Name", "id": "x"},
           "generators": [{"_type": "comprehension",
                           "target": {"_type": "Name", "id": "x"},
                           "iter": {"_type": "Name", "id": "items"},
                           "ifs": [], "is_async": 0}]}
    lc2 = {"_type": "ListComp", "elt": {"_type": "Name", "id": "x"},
           "generators": [{"_type": "comprehension",
                           "target": {"_type": "Name", "id": "x"},
                           "iter": {"_type": "Name", "id": "items"},
                           "ifs": [{"_type": "Compare",
                                    "left": {"_type": "Name", "id": "x"},
                                    "ops": [{"_type": "Gt"}],
                                    "comparators": [{"_type": "Constant", "value": 0}]}],
                           "is_async": 0}]}
    d63 = get_json_distance(lc1, lc2)
    print(f"Test 63 (ListComp add if): {d63:.4f}")
    assert 0.0 < d63 < 0.5

    # Test 64: identical comprehensions
    assert get_json_distance(lc1, lc1) == 0.0
    print(f"Test 64 (ListComp same):   0.0")

    # Test 65: real AST — Lambda
    lam1 = {"_type": "Lambda",
            "args": {"_type": "arguments", "args": [{"_type": "arg", "arg": "x"}],
                     "posonlyargs": [], "kwonlyargs": [], "kw_defaults": [], "defaults": []},
            "body": {"_type": "Name", "id": "x"}}
    lam2 = {"_type": "Lambda",
            "args": {"_type": "arguments", "args": [{"_type": "arg", "arg": "x"}],
                     "posonlyargs": [], "kwonlyargs": [], "kw_defaults": [], "defaults": []},
            "body": {"_type": "BinOp", "left": {"_type": "Name", "id": "x"},
                     "op": {"_type": "Add"}, "right": {"_type": "Constant", "value": 1}}}
    d65 = get_json_distance(lam1, lam2)
    print(f"Test 65 (Lambda body):     {d65:.4f}")
    assert 0.0 < d65 < 0.5

    # Test 66: float precision
    assert get_json_distance(3.14, 3.14) == 0.0
    print(f"Test 66 (float same):      0.0")

    # Test 67: float diff — replace counts both old+new
    d67 = get_json_distance(3.14, 2.71)
    print(f"Test 67 (float diff):      {d67:.4f}")
    assert d67 == 1.0

    # Test 68: large dict identical
    big_d = {f"k{i}": i for i in range(50)}
    assert get_json_distance(big_d, big_d) == 0.0
    print(f"Test 68 (big dict same):   0.0")

    # Test 69: large dict one change
    big_d2 = {f"k{i}": i for i in range(50)}
    big_d2["k25"] = 999
    d69 = get_json_distance(big_d, big_d2)
    print(f"Test 69 (big dict 1 chg):  {d69:.4f}")
    assert 0.0 < d69 < 0.05

    # Test 70: monotonicity on large dict
    big_d3 = {f"k{i}": i for i in range(50)}
    big_d3["k25"] = 999; big_d3["k26"] = 999
    big_d4 = {f"k{i}": i for i in range(50)}
    big_d4["k25"] = 999; big_d4["k26"] = 999; big_d4["k27"] = 999
    d70a = get_json_distance(big_d, big_d2)
    d70b = get_json_distance(big_d, big_d3)
    d70c = get_json_distance(big_d, big_d4)
    print(f"Test 70 (big monotone):    {d70a:.4f} < {d70b:.4f} < {d70c:.4f}")
    assert d70a < d70b < d70c

    # Test 71: real AST — Attribute access chain
    attr1 = {"_type": "Attribute",
             "value": {"_type": "Attribute",
                       "value": {"_type": "Name", "id": "os"},
                       "attr": "path"},
             "attr": "join"}
    attr2 = {"_type": "Attribute",
             "value": {"_type": "Attribute",
                       "value": {"_type": "Name", "id": "os"},
                       "attr": "path"},
             "attr": "exists"}
    d71 = get_json_distance(attr1, attr2)
    print(f"Test 71 (Attr chg leaf):   {d71:.4f}")
    assert 0.0 < d71 < 0.2

    # Test 72: real AST — Attribute chain vs simple Name
    d72 = get_json_distance(attr1, {"_type": "Name", "id": "join"})
    print(f"Test 72 (Attr vs Name):    {d72:.4f}")
    assert d72 > d71

    # Test 73: real AST — Subscript
    sub1 = {"_type": "Subscript", "value": {"_type": "Name", "id": "lst"},
            "slice": {"_type": "Constant", "value": 0}}
    sub2 = {"_type": "Subscript", "value": {"_type": "Name", "id": "lst"},
            "slice": {"_type": "Constant", "value": -1}}
    d73 = get_json_distance(sub1, sub2)
    print(f"Test 73 (Subscript idx):   {d73:.4f}")
    assert 0.0 < d73 < 0.3

    # Test 74: real AST — Slice
    sl1 = {"_type": "Subscript", "value": {"_type": "Name", "id": "lst"},
           "slice": {"_type": "Slice", "lower": {"_type": "Constant", "value": 0},
                     "upper": {"_type": "Constant", "value": 5}}}
    d74 = get_json_distance(sub1, sl1)
    print(f"Test 74 (Index vs Slice):  {d74:.4f}")
    assert d74 > d73, "index→slice is bigger change than index→index"

    # Test 75: real AST — Dict literal
    dict1 = {"_type": "Dict", "keys": [{"_type": "Constant", "value": "a"}],
             "values": [{"_type": "Constant", "value": 1}]}
    dict2 = {"_type": "Dict",
             "keys": [{"_type": "Constant", "value": "a"}, {"_type": "Constant", "value": "b"}],
             "values": [{"_type": "Constant", "value": 1}, {"_type": "Constant", "value": 2}]}
    d75 = get_json_distance(dict1, dict2)
    print(f"Test 75 (Dict add entry):  {d75:.4f}")
    assert 0.0 < d75 < 0.5

    # Test 76: real AST — Compare operators
    cmp1 = {"_type": "Compare", "left": {"_type": "Name", "id": "x"},
            "ops": [{"_type": "Lt"}], "comparators": [{"_type": "Constant", "value": 10}]}
    cmp2 = {"_type": "Compare", "left": {"_type": "Name", "id": "x"},
            "ops": [{"_type": "GtE"}], "comparators": [{"_type": "Constant", "value": 10}]}
    d76 = get_json_distance(cmp1, cmp2)
    print(f"Test 76 (Compare op chg):  {d76:.4f}")
    assert 0.0 < d76 < 0.2

    # Test 77: real AST — BoolOp and/or
    bool1 = {"_type": "BoolOp", "op": {"_type": "And"},
             "values": [{"_type": "Name", "id": "a"}, {"_type": "Name", "id": "b"}]}
    bool2 = {"_type": "BoolOp", "op": {"_type": "Or"},
             "values": [{"_type": "Name", "id": "a"}, {"_type": "Name", "id": "b"}]}
    d77 = get_json_distance(bool1, bool2)
    print(f"Test 77 (BoolOp and/or):   {d77:.4f}")
    assert 0.0 < d77 < 0.2

    # Test 78: real AST — UnaryOp
    un1 = {"_type": "UnaryOp", "op": {"_type": "Not"},
           "operand": {"_type": "Name", "id": "x"}}
    un2 = {"_type": "UnaryOp", "op": {"_type": "USub"},
           "operand": {"_type": "Name", "id": "x"}}
    d78 = get_json_distance(un1, un2)
    print(f"Test 78 (UnaryOp diff):    {d78:.4f}")
    assert 0.0 < d78 < 0.3

    # Test 79: real AST — IfExp (ternary)
    ife1 = {"_type": "IfExp", "test": {"_type": "Name", "id": "cond"},
            "body": {"_type": "Constant", "value": 1},
            "orelse": {"_type": "Constant", "value": 0}}
    ife2 = {"_type": "IfExp", "test": {"_type": "Name", "id": "cond"},
            "body": {"_type": "Constant", "value": "yes"},
            "orelse": {"_type": "Constant", "value": "no"}}
    d79 = get_json_distance(ife1, ife2)
    print(f"Test 79 (IfExp vals):      {d79:.4f}")
    assert 0.0 < d79 < 0.4

    # Test 80: real AST — AugAssign
    aug1 = {"_type": "AugAssign", "target": {"_type": "Name", "id": "x"},
            "op": {"_type": "Add"}, "value": {"_type": "Constant", "value": 1}}
    aug2 = {"_type": "AugAssign", "target": {"_type": "Name", "id": "x"},
            "op": {"_type": "Mult"}, "value": {"_type": "Constant", "value": 2}}
    d80 = get_json_distance(aug1, aug2)
    print(f"Test 80 (AugAssign diff):  {d80:.4f}")
    assert 0.0 < d80 < 0.4

    # Test 81: symmetry batch — 5 random pairs from codebase
    import random
    all_py = []
    for root, _dirs, files in os.walk(codebase):
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py":
                all_py.append(os.path.join(root, f))
    all_py.sort()
    random.seed(81)
    sym_ok = 0
    for _ in range(5):
        fp_a, fp_b = random.sample(all_py, 2)
        with open(fp_a) as fh:
            ta = ast_to_dict(_ast.parse(fh.read()))["body"]
        with open(fp_b) as fh:
            tb = ast_to_dict(_ast.parse(fh.read()))["body"]
        dab = get_json_distance(ta, tb)
        dba = get_json_distance(tb, ta)
        if abs(dab - dba) < 1e-9:
            sym_ok += 1
    print(f"Test 81 (sym batch 5):     {sym_ok}/5")
    assert sym_ok == 5

    # Test 82: all codebase files self-distance == 0
    self_ok = 0
    for fp in all_py[:10]:
        with open(fp) as fh:
            t = ast_to_dict(_ast.parse(fh.read()))["body"]
        if get_json_distance(t, t) == 0.0:
            self_ok += 1
    print(f"Test 82 (self=0 batch):    {self_ok}/10")
    assert self_ok == 10

    # Test 83: real AST — Starred expression
    star1 = {"_type": "Starred", "value": {"_type": "Name", "id": "args"}}
    star2 = {"_type": "Starred", "value": {"_type": "Name", "id": "kwargs"}}
    d83 = get_json_distance(star1, star2)
    print(f"Test 83 (Starred diff):    {d83:.4f}")
    assert 0.0 < d83 < 0.4

    # Test 84: real AST — Yield vs YieldFrom
    y1 = {"_type": "Yield", "value": {"_type": "Constant", "value": 1}}
    y2 = {"_type": "YieldFrom", "value": {"_type": "Name", "id": "gen"}}
    d84 = get_json_distance(y1, y2)
    print(f"Test 84 (Yield types):     {d84:.4f}")
    assert d84 > 0.0

    # Test 85: real AST — Assert with/without msg
    as1 = {"_type": "Assert", "test": {"_type": "Name", "id": "cond"}}
    as2 = {"_type": "Assert", "test": {"_type": "Name", "id": "cond"},
           "msg": {"_type": "Constant", "value": "failed"}}
    d85 = get_json_distance(as1, as2)
    print(f"Test 85 (Assert add msg):  {d85:.4f}")
    assert 0.0 < d85 < 0.5

    # Test 86: real AST — Global/Nonlocal
    gl1 = {"_type": "Global", "names": ["x"]}
    gl2 = {"_type": "Global", "names": ["x", "y"]}
    d86 = get_json_distance(gl1, gl2)
    print(f"Test 86 (Global add var):  {d86:.4f}")
    assert 0.0 < d86 < 0.5

    # Test 87: real AST — Raise with/without cause
    r1 = {"_type": "Raise", "exc": {"_type": "Name", "id": "ValueError"}}
    r2 = {"_type": "Raise", "exc": {"_type": "Name", "id": "ValueError"},
          "cause": {"_type": "Name", "id": "original"}}
    d87 = get_json_distance(r1, r2)
    print(f"Test 87 (Raise add cause): {d87:.4f}")
    assert 0.0 < d87 < 0.5

    # Test 88: real AST — Delete
    del1 = {"_type": "Delete", "targets": [{"_type": "Name", "id": "x"}]}
    del2 = {"_type": "Delete", "targets": [{"_type": "Name", "id": "x"},
                                            {"_type": "Name", "id": "y"}]}
    d88 = get_json_distance(del1, del2)
    print(f"Test 88 (Delete add tgt):  {d88:.4f}")
    assert 0.0 < d88 < 0.5

    # Test 89: real AST — FunctionDef add param
    fndef1 = {"_type": "FunctionDef", "name": "f",
              "args": {"_type": "arguments",
                       "args": [{"_type": "arg", "arg": "x"}],
                       "defaults": [], "posonlyargs": [], "kwonlyargs": [],
                       "kw_defaults": []},
              "body": [{"_type": "Pass"}], "decorator_list": []}
    fndef2 = {"_type": "FunctionDef", "name": "f",
              "args": {"_type": "arguments",
                       "args": [{"_type": "arg", "arg": "x"}, {"_type": "arg", "arg": "y"}],
                       "defaults": [], "posonlyargs": [], "kwonlyargs": [],
                       "kw_defaults": []},
              "body": [{"_type": "Pass"}], "decorator_list": []}
    d89 = get_json_distance(fndef1, fndef2)
    print(f"Test 89 (FnDef add arg):   {d89:.4f}")
    assert 0.0 < d89 < 0.3

    # Test 90: real AST — FunctionDef add return annotation
    fndef3 = {"_type": "FunctionDef", "name": "f",
              "args": {"_type": "arguments",
                       "args": [{"_type": "arg", "arg": "x"}],
                       "defaults": [], "posonlyargs": [], "kwonlyargs": [],
                       "kw_defaults": []},
              "body": [{"_type": "Pass"}], "decorator_list": [],
              "returns": {"_type": "Name", "id": "int"}}
    d90 = get_json_distance(fndef1, fndef3)
    print(f"Test 90 (FnDef add ret):   {d90:.4f}")
    assert 0.0 < d90 < 0.3

    # Test 91: real AST — ClassDef add method
    cls_a = {"_type": "ClassDef", "name": "A", "bases": [], "keywords": [],
             "body": [{"_type": "Pass"}], "decorator_list": []}
    cls_b = {"_type": "ClassDef", "name": "A", "bases": [], "keywords": [],
             "body": [{"_type": "FunctionDef", "name": "__init__",
                       "args": {"_type": "arguments", "args": [{"_type": "arg", "arg": "self"}],
                                "defaults": [], "posonlyargs": [], "kwonlyargs": [],
                                "kw_defaults": []},
                       "body": [{"_type": "Pass"}], "decorator_list": []}],
             "decorator_list": []}
    d91 = get_json_distance(cls_a, cls_b)
    print(f"Test 91 (Class add meth):  {d91:.4f}")
    assert 0.0 < d91 < 0.7

    # Test 92: distance decreases as we add back dropped fields
    # start from fndef1, progressively approach fndef3 (which adds returns annotation)
    d92a = get_json_distance(fndef1, fndef3)
    fndef1b = {**fndef1, "returns": {"_type": "Name", "id": "str"}}
    d92b = get_json_distance(fndef1b, fndef3)
    print(f"Test 92 (converging):      {d92a:.4f} > {d92b:.4f}")
    assert d92a > d92b, "closer AST should have smaller distance"

    # Test 93: real AST — With statement
    with1 = {"_type": "With",
             "items": [{"_type": "withitem",
                        "context_expr": {"_type": "Call", "func": "open",
                                         "args": [{"_type": "Constant", "value": "f.txt"}],
                                         "keywords": []},
                        "optional_vars": {"_type": "Name", "id": "fh"}}],
             "body": [{"_type": "Pass"}]}
    with2 = {"_type": "With",
             "items": [{"_type": "withitem",
                        "context_expr": {"_type": "Call", "func": "open",
                                         "args": [{"_type": "Constant", "value": "g.txt"}],
                                         "keywords": []},
                        "optional_vars": {"_type": "Name", "id": "fh"}}],
             "body": [{"_type": "Pass"}]}
    d93 = get_json_distance(with1, with2)
    print(f"Test 93 (With chg file):   {d93:.4f}")
    assert 0.0 < d93 < 0.2

    # Test 94: real AST — JoinedStr (f-string)
    fs1 = {"_type": "JoinedStr", "values": [
        {"_type": "Constant", "value": "Hello "},
        {"_type": "FormattedValue", "value": {"_type": "Name", "id": "name"},
         "conversion": -1}]}
    fs2 = {"_type": "JoinedStr", "values": [
        {"_type": "Constant", "value": "Hi "},
        {"_type": "FormattedValue", "value": {"_type": "Name", "id": "name"},
         "conversion": -1}]}
    d94 = get_json_distance(fs1, fs2)
    print(f"Test 94 (fstring diff):    {d94:.4f}")
    assert 0.0 < d94 < 0.2

    # Test 95: triangle inequality on real AST nodes
    d95_ab = get_json_distance(for1, while1)
    d95_bc = get_json_distance(while1, try1)
    d95_ac = get_json_distance(for1, try1)
    print(f"Test 95 (tri real AST):    {d95_ac:.4f} <= {d95_ab + d95_bc:.4f}")
    assert d95_ac <= d95_ab + d95_bc + 1e-9

    # Test 96: real roundtrip on 5 more files
    rt_ok = 0
    for fp in all_py[10:15]:
        with open(fp) as fh:
            src = fh.read()
        t_orig = ast_to_dict(_ast.parse(src))["body"]
        jstr = convert_python_to_ast_tag_jsonl(src)
        recs = [json.loads(l) for l in jstr.strip().splitlines()]
        t_recon, _ = convert_ast_tag_jsonl_to_ast_json(recs)
        d = get_json_distance(t_orig, t_recon)
        if d < 0.6:
            rt_ok += 1
    print(f"Test 96 (roundtrip x5):    {rt_ok}/5")
    assert rt_ok == 5

    # Test 97: deeply nested identical structure (10 levels)
    def make_deep(n, leaf):
        if n == 0:
            return leaf
        return {"level": n, "child": make_deep(n - 1, leaf)}
    deep10a = make_deep(10, "same")
    deep10b = make_deep(10, "same")
    assert get_json_distance(deep10a, deep10b) == 0.0
    print(f"Test 97 (deep10 same):     0.0")

    # Test 98: deeply nested single leaf diff
    deep10c = make_deep(10, "diff")
    d98 = get_json_distance(deep10a, deep10c)
    print(f"Test 98 (deep10 1 leaf):   {d98:.4f}")
    assert 0.0 < d98 < 0.1

    # Test 99: real AST — multi-statement body reorder
    body1 = [{"_type": "Assign", "targets": [{"_type": "Name", "id": "x"}],
              "value": {"_type": "Constant", "value": 1}},
             {"_type": "Assign", "targets": [{"_type": "Name", "id": "y"}],
              "value": {"_type": "Constant", "value": 2}},
             {"_type": "Return", "value": {"_type": "Name", "id": "x"}}]
    body2 = [{"_type": "Assign", "targets": [{"_type": "Name", "id": "y"}],
              "value": {"_type": "Constant", "value": 2}},
             {"_type": "Assign", "targets": [{"_type": "Name", "id": "x"}],
              "value": {"_type": "Constant", "value": 1}},
             {"_type": "Return", "value": {"_type": "Name", "id": "x"}}]
    d99 = get_json_distance(body1, body2)
    print(f"Test 99 (body reorder):    {d99:.4f}")
    assert 0.0 < d99 < 0.5

    # Test 100: empty module body vs real module body
    d100 = get_json_distance([], original_json)
    print(f"Test 100 (empty vs mod):   {d100:.4f}")
    assert d100 > 0.3, "empty vs real module should be large"

    # ===================================================================
    # Bug fix tests (101-115)
    # ===================================================================

    # --- Bug #1: _json_safe incomplete (bytes, complex, tuple, set, frozenset) ---

    # Test 101: bytes in AST Constant — previously crashed jsonpatch
    bytes_node1 = {"_type": "Constant", "value": b"hello"}
    bytes_node2 = {"_type": "Constant", "value": b"world"}
    d101 = get_json_distance(bytes_node1, bytes_node2)
    print(f"Test 101 (bytes diff):     {d101:.4f}")
    assert d101 > 0.0, "different bytes should have positive distance"

    # Test 102: bytes identical
    assert get_json_distance(bytes_node1, bytes_node1) == 0.0
    print(f"Test 102 (bytes same):     0.0")

    # Test 103: complex number in AST Constant — previously crashed jsonpatch
    complex_node1 = {"_type": "Constant", "value": 1+2j}
    complex_node2 = {"_type": "Constant", "value": 3+4j}
    d103 = get_json_distance(complex_node1, complex_node2)
    print(f"Test 103 (complex diff):   {d103:.4f}")
    assert d103 > 0.0

    # Test 104: complex identical
    assert get_json_distance(complex_node1, complex_node1) == 0.0
    print(f"Test 104 (complex same):   0.0")

    # Test 105: tuple converted to list
    tup1 = {"items": (1, 2, 3)}
    tup2 = {"items": (1, 2, 4)}
    d105 = get_json_distance(tup1, tup2)
    print(f"Test 105 (tuple diff):     {d105:.4f}")
    assert d105 > 0.0

    # Test 106: set/frozenset handled
    set1 = {"tags": frozenset({"a", "b"})}
    set2 = {"tags": frozenset({"a", "c"})}
    d106 = get_json_distance(set1, set2)
    print(f"Test 106 (frozenset diff): {d106:.4f}")
    assert d106 > 0.0

    # Test 107: real Python AST with bytes literal (end-to-end)
    bytes_ast = ast_to_dict(_ast.parse("x = b'hello'"))
    bytes_ast2 = ast_to_dict(_ast.parse("x = b'world'"))
    d107 = get_json_distance(bytes_ast, bytes_ast2)
    print(f"Test 107 (bytes AST e2e):  {d107:.4f}")
    assert 0.0 < d107 < 1.0

    # Test 108: real Python AST with complex literal
    cplx_ast = ast_to_dict(_ast.parse("x = 1+2j"))
    cplx_ast2 = ast_to_dict(_ast.parse("x = 3+4j"))
    d108 = get_json_distance(cplx_ast, cplx_ast2)
    print(f"Test 108 (complex AST):    {d108:.4f}")
    assert 0.0 < d108 < 1.0

    # --- Bug #2: RFC 6901 JSON Pointer escaping (~0, ~1) ---

    # Test 109: dict key containing '/' — _resolve_pointer must unescape ~1
    slash_doc1 = {"a/b": 1, "c": 2}
    slash_doc2 = {"a/b": 99, "c": 2}
    d109 = get_json_distance(slash_doc1, slash_doc2)
    print(f"Test 109 (key with /):     {d109:.4f}")
    assert 0.0 < d109 < 1.0

    # Test 110: dict key containing '~' — _resolve_pointer must unescape ~0
    tilde_doc1 = {"a~b": 1, "c": 2}
    tilde_doc2 = {"a~b": 99, "c": 2}
    d110 = get_json_distance(tilde_doc1, tilde_doc2)
    print(f"Test 110 (key with ~):     {d110:.4f}")
    assert 0.0 < d110 < 1.0

    # Test 111: _resolve_pointer direct test
    from experience.ast_tag.get_json_distance import _resolve_pointer
    test_doc = {"a/b": {"c~d": 42}}
    assert _resolve_pointer(test_doc, "/a~1b/c~0d") == 42, "RFC 6901 escaping broken"
    print(f"Test 111 (pointer escape):  OK")

    # --- Bug #3: replace counts both old + new value leaves ---

    # Test 112: replace consistency — replace == remove+add in leaf counting
    # Single leaf replace: old(1) + new(1) = 2 leaves touched per direction
    d112 = get_json_distance({"x": "old"}, {"x": "new"})
    print(f"Test 112 (replace=rm+add): {d112:.4f}")
    assert d112 == 1.0, "replacing only leaf should be full distance"

    # Test 113: replace subtree — large old replaced by small new
    big_old = {"x": {"a": 1, "b": 2, "c": 3}}
    small_new = {"x": 0}
    d113 = get_json_distance(big_old, small_new)
    print(f"Test 113 (big→small repl): {d113:.4f}")
    assert d113 > 0.5, "replacing 3-leaf subtree should be large change"

    # Test 114: replace is symmetric with new counting
    d114a = get_json_distance(big_old, small_new)
    d114b = get_json_distance(small_new, big_old)
    print(f"Test 114 (repl symmetry):  {d114a:.4f} vs {d114b:.4f}")
    assert abs(d114a - d114b) < 1e-9

    # --- Bug #4: empty containers count as 1 leaf ---

    # Test 115: _num_leaves on empty containers
    from experience.ast_tag.get_json_distance import _num_leaves
    assert _num_leaves({}) == 1, "empty dict should be 1 leaf"
    assert _num_leaves([]) == 1, "empty list should be 1 leaf"
    assert _num_leaves({"a": []}) == 1, "dict with empty list = 1 leaf"
    assert _num_leaves([[], []]) == 2, "list of 2 empty lists = 2 leaves"
    print(f"Test 115 (empty=1 leaf):    OK")

    # Test 116: distance with empty containers is well-scaled
    d116a = get_json_distance([], [1])
    d116b = get_json_distance([], [1, 2, 3, 4, 5])
    print(f"Test 116 ([] vs [1]={d116a:.4f}, [] vs [1..5]={d116b:.4f})")
    assert d116a < d116b, "adding more items should increase distance"

    # Test 117: empty list vs empty list
    assert get_json_distance([], []) == 0.0
    print(f"Test 117 ([] vs []):        0.0")

    # Test 118: empty dict in larger structure
    d118a = get_json_distance({"a": {}, "b": 1}, {"a": {"x": 1}, "b": 1})
    print(f"Test 118 (fill empty dict): {d118a:.4f}")
    assert 0.0 < d118a < 0.5

    # --- Bug #5: json.dumps/loads instead of copy.deepcopy (perf) ---

    # Test 119: verify correctness on large tree (perf regression test)
    import time
    big_tree = {"nodes": [{"id": i, "children": [{"v": j} for j in range(10)]} for i in range(50)]}
    big_tree2 = json.loads(json.dumps(big_tree))
    big_tree2["nodes"][25]["children"][5]["v"] = 999
    t0 = time.time()
    d119 = get_json_distance(big_tree, big_tree2)
    elapsed = time.time() - t0
    print(f"Test 119 (big tree perf):  d={d119:.4f} in {elapsed:.3f}s")
    assert 0.0 < d119 < 0.05
    assert elapsed < 10.0, f"too slow: {elapsed:.1f}s"

    # Test 120: _json_safe correctness — tuple, set, bytes, complex all roundtrip
    from experience.ast_tag.get_json_distance import _json_safe
    safe = _json_safe({"t": (1, 2), "s": {3, 1, 2}, "b": b"x", "c": 1+2j, "e": ...})
    assert safe["t"] == [1, 2], f"tuple not converted: {safe['t']}"
    assert safe["s"] == [1, 2, 3], f"set not sorted: {safe['s']}"
    assert safe["b"] == "b'x'", f"bytes not repr'd: {safe['b']}"
    assert safe["c"] == "(1+2j)", f"complex not str'd: {safe['c']}"
    assert safe["e"] == "...", f"ellipsis not converted: {safe['e']}"
    print(f"Test 120 (_json_safe all):  OK")

    print(f"\nAll 120 tests passed.")
