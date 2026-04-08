"""Convert AST Tag JSONL relations back to AST JSON tree.

Generated from convert_ast_tag_jsonl_to_ast_json.viba.

convert_ast_tag_jsonl_to_ast_json :=
    ($json list[PythonJson], $inner_path_to_tmp_var_name dict[str, str])
    <- list[AstTagRelation[Python]]
    # inline
    <- {
        1. reconstruct ast in json format
        2. all tmp_var started from '$' should be inlined, i.e. make nested ast tree
        3. the name convention of inner_path is the same as jsondiff path
    }
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Relation index — group flat relations by (owner_tag, relation_tag)
# ---------------------------------------------------------------------------

class _RelationIndex:
    """Index flat relations by owner_tag for fast lookup."""

    def __init__(self, records: List[Dict]):
        # forward: (owner_tag, relation_tag) -> [(member_tag, member_order_value, line)]
        self._fwd: Dict[Tuple[str, str], List[Tuple[str, int, int]]] = {}
        # reverse: member_tag -> [(owner_tag, relation_tag, member_order_value, line)]
        self._rev: Dict[str, List[Tuple[str, str, int, int]]] = {}
        # all relation_tags for an owner
        self._owner_tags: Dict[str, set] = {}

        for r in records:
            owner = r["owner_tag"]
            rel = r["relation_tag"]
            member = r["member_tag"]
            order = r["member_order_value"]
            line = r.get("line", 0)

            key = (owner, rel)
            self._fwd.setdefault(key, []).append((member, order, line))
            self._rev.setdefault(member, []).append((owner, rel, order, line))
            self._owner_tags.setdefault(owner, set()).add(rel)

        # sort by member_order_value
        for key in self._fwd:
            self._fwd[key].sort(key=lambda x: x[1])

    def get(self, owner: str, rel_tag: str) -> List[str]:
        """Get member_tags for (owner, rel_tag) sorted by order."""
        return [m for m, _, _ in self._fwd.get((owner, rel_tag), [])]

    def get_with_line(self, owner: str, rel_tag: str) -> List[Tuple[str, int]]:
        """Get (member_tag, line) pairs sorted by order."""
        return [(m, ln) for m, _, ln in self._fwd.get((owner, rel_tag), [])]

    def tags(self, owner: str) -> set:
        """All relation_tags for owner."""
        return self._owner_tags.get(owner, set())

    def get_line(self, owner: str) -> int:
        """Best guess line number for an owner (min line from its relations)."""
        tags = self._owner_tags.get(owner, set())
        min_ln = 0
        for tag in tags:
            for _, _, ln in self._fwd.get((owner, tag), []):
                if ln > 0 and (min_ln == 0 or ln < min_ln):
                    min_ln = ln
        return min_ln


# ---------------------------------------------------------------------------
# Operator symbol -> AST type mapping (inverse of _get_op_symbol)
# ---------------------------------------------------------------------------

_BINOP_MAP = {
    "+": "Add", "-": "Sub", "*": "Mult", "/": "Div", "//": "FloorDiv",
    "%": "Mod", "**": "Pow", "<<": "LShift", ">>": "RShift",
    "|": "BitOr", "^": "BitXor", "&": "BitAnd", "@": "MatMult",
}
_UNARYOP_MAP = {
    "+": "UAdd", "-": "USub", "not": "Not", "~": "Invert",
}
_CMPOP_MAP = {
    "==": "Eq", "!=": "NotEq", "<": "Lt", "<=": "LtE",
    ">": "Gt", ">=": "GtE", "is": "Is", "is_not": "IsNot",
    "in": "In", "not_in": "NotIn",
}
_BOOLOP_MAP = {
    "and": "And", "or": "Or",
}
_AUGOP_MAP = {
    "+=": "Add", "-=": "Sub", "*=": "Mult", "/=": "Div", "//=": "FloorDiv",
    "%=": "Mod", "**=": "Pow", "<<=": "LShift", ">>=": "RShift",
    "|=": "BitOr", "^=": "BitXor", "&=": "BitAnd", "@=": "MatMult",
    # Also accept bare operator symbols since the forward converter stores those
    "+": "Add", "-": "Sub", "*": "Mult", "/": "Div", "//": "FloorDiv",
    "%": "Mod", "**": "Pow", "<<": "LShift", ">>": "RShift",
    "|": "BitOr", "^": "BitXor", "&": "BitAnd", "@": "MatMult",
}


# ---------------------------------------------------------------------------
# Symbol -> AST JSON node reconstruction
# ---------------------------------------------------------------------------

def _is_tmp_var(sym: str) -> bool:
    return sym.startswith("$")


def _tmp_var_prefix(sym: str) -> str:
    """Extract prefix from $prefix_N -> prefix."""
    m = re.match(r"^\$(.+?)_\d+$", sym)
    return m.group(1) if m else ""


def _symbol_to_node(sym: str, idx: _RelationIndex, path: str,
                     path_map: Dict[str, str]) -> Any:
    """Convert a symbol back to an AST JSON node. Recursive for tmp vars."""
    if not _is_tmp_var(sym):
        # leaf symbol — figure out what kind of node it is
        if sym == "None":
            return {"_type": "Constant", "value": None}
        if sym == "True":
            return {"_type": "Constant", "value": True}
        if sym == "False":
            return {"_type": "Constant", "value": False}
        if sym == "...":
            return {"_type": "Constant", "value": ...}
        # Check if it's a repr'd constant (starts with quote, number, etc.)
        if sym.startswith("'") or sym.startswith('"'):
            try:
                return {"_type": "Constant", "value": eval(sym)}
            except Exception:
                return {"_type": "Name", "id": sym}
        # Try numeric
        try:
            val = int(sym)
            return {"_type": "Constant", "value": val}
        except ValueError:
            pass
        try:
            val = float(sym)
            return {"_type": "Constant", "value": val}
        except ValueError:
            pass
        # Dotted name -> Attribute chain
        if "." in sym:
            parts = sym.split(".")
            node = {"_type": "Name", "id": parts[0]}
            for attr in parts[1:]:
                node = {"_type": "Attribute", "value": node, "attr": attr}
            return node
        # Simple name
        return {"_type": "Name", "id": sym}

    # tmp var — reconstruct from relations
    path_map[path] = sym
    prefix = _tmp_var_prefix(sym)
    tags = idx.tags(sym)
    ln = idx.get_line(sym)

    def child_node(member: str, child_path: str) -> Any:
        return _symbol_to_node(member, idx, child_path, path_map)

    def make_node(type_name: str, **fields) -> Dict:
        d: Dict[str, Any] = {"_type": type_name}
        if ln > 0:
            d["_lineno"] = ln
        d.update(fields)
        return d

    # Call
    if prefix == "call" or "calls" in tags:
        func_members = idx.get(sym, "calls")
        func_node = child_node(func_members[0], f"{path}/func") if func_members else None
        args = []
        for i, m in enumerate(idx.get(sym, "call_pos_arg")):
            if m.startswith("*"):
                inner = child_node(m[1:], f"{path}/args/{i}/value")
                args.append({"_type": "Starred", "value": inner})
            else:
                args.append(child_node(m, f"{path}/args/{i}"))
        keywords = []
        kw_members = idx.get(sym, "keyword_arg")
        # keyword_arg comes in pairs: name, value
        ki = 0
        for j in range(0, len(kw_members), 2):
            if j + 1 < len(kw_members):
                kw_name = kw_members[j]
                kw_val = child_node(kw_members[j + 1], f"{path}/keywords/{ki}/value")
                keywords.append({"_type": "keyword", "arg": kw_name, "value": kw_val})
                ki += 1
        for m in idx.get(sym, "double_star_arg"):
            val = child_node(m, f"{path}/keywords/{ki}/value")
            keywords.append({"_type": "keyword", "arg": None, "value": val})
            ki += 1
        return make_node("Call", func=func_node, args=args, keywords=keywords)

    # Attribute
    if prefix == "attr" or ("attr_value" in tags and "attr_name" in tags):
        value_members = idx.get(sym, "attr_value")
        name_members = idx.get(sym, "attr_name")
        value_node = child_node(value_members[0], f"{path}/value") if value_members else None
        attr_name = name_members[0] if name_members else ""
        return make_node("Attribute", value=value_node, attr=attr_name)

    # Subscript
    if prefix == "subscript" or "subscript_value" in tags:
        val_members = idx.get(sym, "subscript_value")
        slice_members = idx.get(sym, "subscript")
        value_node = child_node(val_members[0], f"{path}/value") if val_members else None
        slice_node = child_node(slice_members[0], f"{path}/slice") if slice_members else None
        return make_node("Subscript", value=value_node, slice=slice_node)

    # Slice
    if prefix == "slice" or "slice_marker" in tags or "slice_lower" in tags or "slice_upper" in tags or "slice_step" in tags:
        lower = None
        upper = None
        step = None
        lo = idx.get(sym, "slice_lower")
        if lo:
            lower = child_node(lo[0], f"{path}/lower")
        up = idx.get(sym, "slice_upper")
        if up:
            upper = child_node(up[0], f"{path}/upper")
        st = idx.get(sym, "slice_step")
        if st:
            step = child_node(st[0], f"{path}/step")
        return make_node("Slice", lower=lower, upper=upper, step=step)

    # BinOp
    if prefix == "binop" or "bin_op" in tags:
        op_sym = idx.get(sym, "bin_op")[0] if idx.get(sym, "bin_op") else "+"
        left_sym = idx.get(sym, "bin_op_left")[0] if idx.get(sym, "bin_op_left") else "0"
        right_sym = idx.get(sym, "bin_op_right")[0] if idx.get(sym, "bin_op_right") else "0"
        return make_node("BinOp",
                         left=child_node(left_sym, f"{path}/left"),
                         op={"_type": _BINOP_MAP.get(op_sym, "Add")},
                         right=child_node(right_sym, f"{path}/right"))

    # UnaryOp
    if prefix == "unaryop" or "unary_op" in tags:
        op_sym = idx.get(sym, "unary_op")[0] if idx.get(sym, "unary_op") else "~"
        operand_sym = idx.get(sym, "unary_op_operand")[0] if idx.get(sym, "unary_op_operand") else "0"
        return make_node("UnaryOp",
                         op={"_type": _UNARYOP_MAP.get(op_sym, "Invert")},
                         operand=child_node(operand_sym, f"{path}/operand"))

    # Compare
    if prefix == "compare" or "compare_left" in tags:
        left_sym = idx.get(sym, "compare_left")[0] if idx.get(sym, "compare_left") else "0"
        ops = [{"_type": _CMPOP_MAP.get(o, "Eq")} for o in idx.get(sym, "compare_op")]
        comparators = [child_node(c, f"{path}/comparators/{i}")
                       for i, c in enumerate(idx.get(sym, "compare_right"))]
        return make_node("Compare",
                         left=child_node(left_sym, f"{path}/left"),
                         ops=ops, comparators=comparators)

    # BoolOp
    if prefix == "boolop" or "bool_op" in tags:
        op_sym = idx.get(sym, "bool_op")[0] if idx.get(sym, "bool_op") else "and"
        values = [child_node(v, f"{path}/values/{i}")
                  for i, v in enumerate(idx.get(sym, "bool_op_operand"))]
        return make_node("BoolOp",
                         op={"_type": _BOOLOP_MAP.get(op_sym, "And")},
                         values=values)

    # IfExp (ternary)
    if prefix == "ifexp" or "if_expr_test" in tags:
        test_sym = idx.get(sym, "if_expr_test")[0] if idx.get(sym, "if_expr_test") else "True"
        body_sym = idx.get(sym, "if_expr_body")[0] if idx.get(sym, "if_expr_body") else "None"
        else_sym = idx.get(sym, "if_expr_else")[0] if idx.get(sym, "if_expr_else") else "None"
        return make_node("IfExp",
                         test=child_node(test_sym, f"{path}/test"),
                         body=child_node(body_sym, f"{path}/body"),
                         orelse=child_node(else_sym, f"{path}/orelse"))

    # Lambda
    if prefix == "lambda" or "lambda_body" in tags:
        args_node = _reconstruct_args(sym, idx, path, path_map)
        body_members = idx.get(sym, "lambda_body")
        body_node = child_node(body_members[0], f"{path}/body") if body_members else None
        return make_node("Lambda", args=args_node, body=body_node)

    # Await
    if prefix == "await" or "await_value" in tags:
        val_sym = idx.get(sym, "await_value")[0] if idx.get(sym, "await_value") else "None"
        return make_node("Await", value=child_node(val_sym, f"{path}/value"))

    # Yield
    if prefix == "yield" or "yields" in tags:
        val_sym = idx.get(sym, "yields")
        if val_sym and val_sym[0] != "None":
            return make_node("Yield", value=child_node(val_sym[0], f"{path}/value"))
        return make_node("Yield", value=None)

    # YieldFrom
    if prefix == "yieldfrom" or "yields_from" in tags:
        val_sym = idx.get(sym, "yields_from")[0] if idx.get(sym, "yields_from") else "None"
        return make_node("YieldFrom", value=child_node(val_sym, f"{path}/value"))

    # Comprehension — check before Dict/Tuple/List/Set because comps also have
    # dict_literal/list_literal/set_literal/tuple_literal tags
    if prefix == "comp" or "comprehension_body" in tags:
        lit_tags = tags & {"list_literal", "set_literal", "dict_literal", "tuple_literal"}
        lit_tag = next(iter(lit_tags)) if lit_tags else "list_literal"
        lit_val = idx.get(sym, lit_tag)[0] if idx.get(sym, lit_tag) else "ListComp"
        type_map = {
            "ListComp": "ListComp", "SetComp": "SetComp",
            "DictComp": "DictComp", "GeneratorExp": "GeneratorExp",
        }
        comp_type = type_map.get(lit_val, "ListComp")

        body_members = idx.get(sym, "comprehension_body")
        targets = idx.get(sym, "comprehension_target")
        iters = idx.get(sym, "comprehension_iter")
        ifs = idx.get(sym, "comprehension_if")

        generators = []
        for gi in range(len(targets)):
            gen = {
                "_type": "comprehension",
                "target": child_node(targets[gi], f"{path}/generators/{gi}/target"),
                "iter": child_node(iters[gi], f"{path}/generators/{gi}/iter") if gi < len(iters) else None,
                "ifs": [],
                "is_async": 0,
            }
            generators.append(gen)
        # attach ifs to last generator (simplification)
        if generators and ifs:
            generators[-1]["ifs"] = [child_node(c, f"{path}/generators/{len(generators)-1}/ifs/{i}")
                                     for i, c in enumerate(ifs)]

        node = make_node(comp_type, generators=generators)
        if comp_type == "DictComp" and len(body_members) >= 2:
            node["key"] = child_node(body_members[0], f"{path}/key")
            node["value"] = child_node(body_members[1], f"{path}/value")
        elif body_members:
            node["elt"] = child_node(body_members[0], f"{path}/elt")
        return node

    # Dict literal
    if prefix == "dict" or "dict_literal" in tags:
        keys = [child_node(k, f"{path}/keys/{i}") for i, k in enumerate(idx.get(sym, "dict_key"))]
        values = [child_node(v, f"{path}/values/{i}") for i, v in enumerate(idx.get(sym, "dict_value"))]
        # double_star_arg -> None key
        for ds in idx.get(sym, "double_star_arg"):
            keys.append(None)
            values.append(child_node(ds, f"{path}/values/{len(values)}"))
        return make_node("Dict", keys=keys, values=values)

    # Tuple / List / Set
    if prefix in ("tuple", "list", "set"):
        type_map = {"tuple": "Tuple", "list": "List", "set": "Set"}
        elements = [child_node(e, f"{path}/elts/{i}")
                    for i, e in enumerate(idx.get(sym, "collection_element"))]
        return make_node(type_map[prefix], elts=elements)

    # f-string
    if prefix == "fstring" or "fstring_elem" in tags:
        values = []
        for i, elem in enumerate(idx.get(sym, "fstring_elem")):
            if elem.startswith("lit:"):
                text = elem[4:]
                values.append({"_type": "Constant", "value": text})
            elif elem.startswith("val:"):
                rest = elem[4:]
                # Parse val:inner_sym|!s:%fmt
                parts = rest.split("|", 1)
                inner_sym = parts[0]
                conversion = -1
                format_spec = None
                if len(parts) > 1:
                    suffix = parts[1]
                    if "!s" in suffix:
                        conversion = ord("s")
                    elif "!r" in suffix:
                        conversion = ord("r")
                    elif "!a" in suffix:
                        conversion = ord("a")
                    if ":" in suffix:
                        fmt = suffix.split(":", 1)[1]
                        if fmt:
                            format_spec = {
                                "_type": "JoinedStr",
                                "values": [{"_type": "Constant", "value": fmt}],
                            }
                fv = {
                    "_type": "FormattedValue",
                    "value": child_node(inner_sym, f"{path}/values/{i}/value"),
                    "conversion": conversion,
                    "format_spec": format_spec,
                }
                values.append(fv)
        return make_node("JoinedStr", values=values)

    # Starred
    if prefix == "starred" or "starred_value" in tags:
        val_sym = idx.get(sym, "starred_value")[0] if idx.get(sym, "starred_value") else "None"
        return make_node("Starred", value=child_node(val_sym, f"{path}/value"))

    # Walrus (NamedExpr)
    if prefix == "walrus" or "walrus" in tags:
        members = idx.get(sym, "walrus")
        target = child_node(members[0], f"{path}/target") if members else None
        value = child_node(members[1], f"{path}/value") if len(members) > 1 else None
        return make_node("NamedExpr", target=target, value=value)

    # Docstring tmp var (has text_content)
    if prefix == "docstring" or "text_content" in tags:
        text_members = idx.get(sym, "text_content")
        if text_members:
            return {"_type": "Constant", "value": text_members[0]}
        return {"_type": "Constant", "value": ""}

    # Fallback — return Name with the symbol
    return {"_type": "Name", "id": sym}


# ---------------------------------------------------------------------------
# Reconstruct function args from relations
# ---------------------------------------------------------------------------

def _reconstruct_args(owner: str, idx: _RelationIndex, path: str,
                       path_map: Dict[str, str]) -> Dict:
    """Reconstruct an 'arguments' node from param/star_param/double_star_param relations."""
    args_node: Dict[str, Any] = {
        "_type": "arguments",
        "posonlyargs": [],
        "args": [],
        "vararg": None,
        "kwonlyargs": [],
        "kw_defaults": [],
        "kwarg": None,
        "defaults": [],
    }

    params = idx.get(owner, "param")
    param_anns = idx.get(owner, "param_annotation")
    param_defaults = idx.get(owner, "param_default")

    # Build annotation map: param_name -> annotation_sym
    ann_map: Dict[str, str] = {}
    for i in range(0, len(param_anns), 2):
        if i + 1 < len(param_anns):
            ann_map[param_anns[i]] = param_anns[i + 1]

    # Build default map: param_name -> default_sym
    def_map: Dict[str, str] = {}
    for i in range(0, len(param_defaults), 2):
        if i + 1 < len(param_defaults):
            def_map[param_defaults[i]] = param_defaults[i + 1]

    for i, pname in enumerate(params):
        arg = {"_type": "arg", "arg": pname, "annotation": None}
        if pname in ann_map:
            arg["annotation"] = _symbol_to_node(ann_map[pname], idx,
                                                 f"{path}/args/args/{i}/annotation", path_map)
        args_node["args"].append(arg)
        if pname in def_map:
            args_node["defaults"].append(
                _symbol_to_node(def_map[pname], idx,
                                f"{path}/args/defaults/{len(args_node['defaults'])}", path_map))

    star = idx.get(owner, "star_param")
    if star:
        va = {"_type": "arg", "arg": star[0], "annotation": None}
        if star[0] in ann_map:
            va["annotation"] = _symbol_to_node(ann_map[star[0]], idx,
                                                f"{path}/args/vararg/annotation", path_map)
        args_node["vararg"] = va

    dstar = idx.get(owner, "double_star_param")
    if dstar:
        ka = {"_type": "arg", "arg": dstar[0], "annotation": None}
        if dstar[0] in ann_map:
            ka["annotation"] = _symbol_to_node(ann_map[dstar[0]], idx,
                                                f"{path}/args/kwarg/annotation", path_map)
        args_node["kwarg"] = ka

    return args_node


# ---------------------------------------------------------------------------
# Reconstruct statement from owner symbol
# ---------------------------------------------------------------------------

def _reconstruct_stmt(owner: str, idx: _RelationIndex, path: str,
                       path_map: Dict[str, str]) -> Dict:
    """Reconstruct a statement AST JSON node from its owner_tag's relations."""
    tags = idx.tags(owner)
    ln = idx.get_line(owner)

    def child_node(member: str, child_path: str) -> Any:
        return _symbol_to_node(member, idx, child_path, path_map)

    def child_stmt(member: str, child_path: str) -> Dict:
        """Reconstruct a child statement (not expression) from its symbol."""
        return _reconstruct_stmt(member, idx, child_path, path_map)

    def make_node(type_name: str, **fields) -> Dict:
        d: Dict[str, Any] = {"_type": type_name}
        if ln > 0:
            d["_lineno"] = ln
        d.update(fields)
        return d

    path_map[path] = owner

    # ClassDef — check before FunctionDef because both have "defines" + "stmt_seq",
    # but only ClassDef has "bases" or a $classdef_ prefix
    if "defines" in tags and ("bases" in tags or owner.startswith("$classdef_")):
        name = idx.get(owner, "defines")[0] if idx.get(owner, "defines") else owner
        bases = [child_node(b, f"{path}/bases/{i}")
                 for i, b in enumerate(idx.get(owner, "bases"))]
        keywords = []
        mc = idx.get(owner, "metaclass")
        if mc:
            keywords.append({"_type": "keyword", "arg": "metaclass",
                             "value": child_node(mc[0], f"{path}/keywords/0/value")})

        decorators = []
        rev = idx._rev.get(name, [])
        for dec_owner, dec_rel, _, _ in rev:
            if dec_rel == "decorates":
                decorators.append(child_node(dec_owner, f"{path}/decorator_list/{len(decorators)}"))

        body = _reconstruct_body(owner, idx, path, path_map)
        return make_node("ClassDef", name=name, bases=bases, keywords=keywords,
                         body=body, decorator_list=decorators)

    # FunctionDef / AsyncFunctionDef
    if "defines" in tags and ("param" in tags or "star_param" in tags or
                               "ellipsis_body" in tags or "stmt_seq" in tags or
                               "docstring" in tags):
        is_async = "async_def" in tags
        name = idx.get(owner, "defines")[0] if idx.get(owner, "defines") else owner
        args_node = _reconstruct_args(owner, idx, path, path_map)

        # Decorators
        decorators = []
        rev = idx._rev.get(name, [])
        for dec_owner, dec_rel, _, _ in rev:
            if dec_rel == "decorates":
                decorators.append(child_node(dec_owner, f"{path}/decorator_list/{len(decorators)}"))

        # Return annotation
        returns = None
        ret_members = idx.get(owner, "returns")
        if ret_members:
            returns = child_node(ret_members[0], f"{path}/returns")

        # Body
        body = _reconstruct_body(owner, idx, path, path_map)

        node = make_node("AsyncFunctionDef" if is_async else "FunctionDef",
                         name=name, args=args_node, body=body,
                         decorator_list=decorators, returns=returns)
        return node

    # Assign
    if "assigns" in tags and "aug_op" not in tags and "annotation" not in tags:
        members = idx.get(owner, "assigns")
        # members come in pairs: target, value (or multiple targets + value)
        if len(members) >= 2:
            # last member is the value, rest are target+value pairs
            # Actually: for each target, we get target_sym, value_sym
            # So members = [tgt1, val, tgt2, val, ...]
            # But usually just [tgt, val]
            targets = [child_node(members[i], f"{path}/targets/{i//2}")
                       for i in range(0, len(members) - 1, 2)]
            value = child_node(members[-1], f"{path}/value")
            return make_node("Assign", targets=targets, value=value)

    # AugAssign
    if "aug_assigns" in tags:
        members = idx.get(owner, "aug_assigns")
        op_sym = idx.get(owner, "aug_op")[0] if idx.get(owner, "aug_op") else "+"
        target = child_node(members[0], f"{path}/target") if members else None
        value = child_node(members[1], f"{path}/value") if len(members) > 1 else None
        return make_node("AugAssign", target=target,
                         op={"_type": _AUGOP_MAP.get(op_sym, "Add")},
                         value=value)

    # AnnAssign
    if "annotation" in tags:
        ann_members = idx.get(owner, "annotation")
        target = child_node(ann_members[0], f"{path}/target") if ann_members else None
        annotation = child_node(ann_members[1], f"{path}/annotation") if len(ann_members) > 1 else None
        value = None
        if "assigns" in tags:
            val_members = idx.get(owner, "assigns")
            if val_members:
                value = child_node(val_members[0], f"{path}/value")
        return make_node("AnnAssign", target=target, annotation=annotation,
                         value=value, simple=1)

    # ImportFrom — check before Import because $importfrom_ starts with $import_
    if "imports" in tags and owner.startswith("$importfrom_"):
        names = idx.get(owner, "imports")
        aliases_list = idx.get(owner, "aliases")
        # first imports member is the module, rest are names
        module = names[0] if names else ""
        result_names = []
        for i, n in enumerate(names[1:]):
            alias = {"_type": "alias", "name": n, "asname": None}
            if i < len(aliases_list):
                alias["asname"] = aliases_list[i]
            result_names.append(alias)
        return make_node("ImportFrom", module=module if module else None,
                         names=result_names, level=0)

    # Import
    if "imports" in tags and owner.startswith("$import_"):
        names = idx.get(owner, "imports")
        aliases_list = idx.get(owner, "aliases")
        result_names = []
        for i, n in enumerate(names):
            alias = {"_type": "alias", "name": n, "asname": None}
            if i < len(aliases_list):
                alias["asname"] = aliases_list[i]
            result_names.append(alias)
        return make_node("Import", names=result_names)

    # Return
    if "returns" in tags and "defines" not in tags:
        ret_members = idx.get(owner, "returns")
        value = child_node(ret_members[0], f"{path}/value") if ret_members else None
        return make_node("Return", value=value)

    # Bare return
    if "bare_return" in tags:
        return make_node("Return", value=None)

    # Expr (expression statement)
    if "expr_stmt" in tags:
        expr_members = idx.get(owner, "expr_stmt")
        value = child_node(expr_members[0], f"{path}/value") if expr_members else None
        return make_node("Expr", value=value)

    # Expr with docstring
    if "docstring" in tags and "defines" not in tags and "stmt_seq" not in tags:
        doc_members = idx.get(owner, "docstring")
        if doc_members:
            doc_sym = doc_members[0]
            text_members = idx.get(doc_sym, "text_content")
            text = text_members[0] if text_members else ""
            return make_node("Expr", value={"_type": "Constant", "value": text})

    # Yield statement
    if "yields" in tags and "defines" not in tags:
        val_members = idx.get(owner, "yields")
        val = child_node(val_members[0], f"{path}/value/value") if val_members and val_members[0] != "None" else None
        return make_node("Expr", value={"_type": "Yield", "value": val})

    # YieldFrom statement
    if "yields_from" in tags and "defines" not in tags:
        val_members = idx.get(owner, "yields_from")
        val = child_node(val_members[0], f"{path}/value/value") if val_members else None
        return make_node("Expr", value={"_type": "YieldFrom", "value": val})

    # If
    if "if_test" in tags:
        test_sym = idx.get(owner, "if_test")[0]
        body = [child_stmt(s, f"{path}/body/{i}") for i, s in enumerate(idx.get(owner, "if_body"))]
        orelse = [child_stmt(s, f"{path}/orelse/{i}") for i, s in enumerate(idx.get(owner, "else_body"))]
        return make_node("If", test=child_node(test_sym, f"{path}/test"),
                         body=body, orelse=orelse)

    # For / AsyncFor
    if "for_target" in tags:
        is_async = "async_for" in tags
        target_sym = idx.get(owner, "for_target")[0]
        iter_sym = idx.get(owner, "for_iter")[0]
        body = [child_stmt(s, f"{path}/body/{i}") for i, s in enumerate(idx.get(owner, "for_body"))]
        orelse = [child_stmt(s, f"{path}/orelse/{i}") for i, s in enumerate(idx.get(owner, "else_body"))]
        return make_node("AsyncFor" if is_async else "For",
                         target=child_node(target_sym, f"{path}/target"),
                         iter=child_node(iter_sym, f"{path}/iter"),
                         body=body, orelse=orelse)

    # While
    if "while_test" in tags:
        test_sym = idx.get(owner, "while_test")[0]
        body = [child_stmt(s, f"{path}/body/{i}") for i, s in enumerate(idx.get(owner, "while_body"))]
        orelse = [child_stmt(s, f"{path}/orelse/{i}") for i, s in enumerate(idx.get(owner, "else_body"))]
        return make_node("While", test=child_node(test_sym, f"{path}/test"),
                         body=body, orelse=orelse)

    # With / AsyncWith
    if "with_context" in tags:
        is_async = "async_with" in tags
        items = []
        for i, ctx_sym in enumerate(idx.get(owner, "with_context")):
            item = {
                "_type": "withitem",
                "context_expr": child_node(ctx_sym, f"{path}/items/{i}/context_expr"),
                "optional_vars": None,
            }
            as_members = idx.get(ctx_sym, "with_as")
            if as_members:
                item["optional_vars"] = child_node(as_members[0], f"{path}/items/{i}/optional_vars")
            items.append(item)
        body = [child_stmt(s, f"{path}/body/{i}") for i, s in enumerate(idx.get(owner, "with_body"))]
        return make_node("AsyncWith" if is_async else "With", items=items, body=body)

    # Try
    if "try_start" in tags:
        body = [child_stmt(s, f"{path}/body/{i}") for i, s in enumerate(idx.get(owner, "try_body"))]
        handlers = []
        handle_types = idx.get(owner, "handles")
        except_bodies = idx.get(owner, "except_body")
        # Group except bodies per handler — simplified: one handler per handles entry
        for i, exc_sym in enumerate(handle_types):
            as_members = idx.get(exc_sym, "except_as")
            handler = {
                "_type": "ExceptHandler",
                "type": child_node(exc_sym, f"{path}/handlers/{i}/type"),
                "name": as_members[0] if as_members else None,
                "body": [],
            }
            handlers.append(handler)
        # Assign except_body statements to the last handler (simplification)
        if handlers and except_bodies:
            handlers[-1]["body"] = [child_stmt(s, f"{path}/handlers/{len(handlers)-1}/body/{i}")
                                     for i, s in enumerate(except_bodies)]
        elif except_bodies:
            # bare except
            handler = {
                "_type": "ExceptHandler",
                "type": None, "name": None,
                "body": [child_stmt(s, f"{path}/handlers/0/body/{i}")
                         for i, s in enumerate(except_bodies)],
            }
            handlers.append(handler)
        orelse = [child_stmt(s, f"{path}/orelse/{i}") for i, s in enumerate(idx.get(owner, "try_else"))]
        finalbody = [child_stmt(s, f"{path}/finalbody/{i}") for i, s in enumerate(idx.get(owner, "finally_body"))]
        return make_node("Try", body=body, handlers=handlers, orelse=orelse, finalbody=finalbody)

    # Raise
    if "raises" in tags and "raises_from" not in tags:
        exc_sym = idx.get(owner, "raises")[0] if idx.get(owner, "raises") else None
        node = make_node("Raise", exc=child_node(exc_sym, f"{path}/exc") if exc_sym else None, cause=None)
        return node
    if "raises" in tags:
        exc_sym = idx.get(owner, "raises")[0]
        cause_members = idx.get(exc_sym, "raises_from")
        cause = child_node(cause_members[0], f"{path}/cause") if cause_members else None
        return make_node("Raise", exc=child_node(exc_sym, f"{path}/exc"), cause=cause)

    # Pass / Break / Continue
    if "pass_stmt" in tags:
        return make_node("Pass")
    if "break_stmt" in tags:
        return make_node("Break")
    if "continue_stmt" in tags:
        return make_node("Continue")

    # Global / Nonlocal
    if "global_decl" in tags:
        return make_node("Global", names=idx.get(owner, "global_decl"))
    if "nonlocal_decl" in tags:
        return make_node("Nonlocal", names=idx.get(owner, "nonlocal_decl"))

    # Assert
    if "assert_test" in tags:
        test_sym = idx.get(owner, "assert_test")[0]
        msg = None
        msg_members = idx.get(owner, "assert_msg")
        if msg_members:
            msg = child_node(msg_members[0], f"{path}/msg")
        return make_node("Assert", test=child_node(test_sym, f"{path}/test"), msg=msg)

    # Delete
    if "del_target" in tags:
        targets = [child_node(t, f"{path}/targets/{i}")
                   for i, t in enumerate(idx.get(owner, "del_target"))]
        return make_node("Delete", targets=targets)

    # Ellipsis body (standalone)
    if "ellipsis_body" in tags and "defines" not in tags:
        return make_node("Expr", value={"_type": "Constant", "value": ...})

    # Fallback: treat as expression node
    return _symbol_to_node(owner, idx, path, path_map)


# ---------------------------------------------------------------------------
# Body reconstruction from stmt_seq / contains
# ---------------------------------------------------------------------------

def _reconstruct_body(owner: str, idx: _RelationIndex, path: str,
                       path_map: Dict[str, str]) -> List[Dict]:
    """Reconstruct body statements from stmt_seq/contains/docstring relations."""
    body = []

    # Ellipsis body
    if "ellipsis_body" in idx.tags(owner):
        body.append({"_type": "Expr", "value": {"_type": "Constant", "value": ...}})
        return body

    # Use stmt_seq for ordering (stmt_seq already includes docstring Expr if present)
    stmt_seq = idx.get(owner, "stmt_seq")
    if stmt_seq:
        for i, seq_entry in enumerate(stmt_seq):
            # format: "N:$sym"
            parts = seq_entry.split(":", 1)
            if len(parts) == 2:
                stmt_sym = parts[1]
            else:
                stmt_sym = seq_entry
            stmt_node = _reconstruct_stmt(stmt_sym, idx, f"{path}/body/{i}", path_map)
            body.append(stmt_node)
        return body

    # Fallback: use contains
    contains = idx.get(owner, "contains")
    if contains:
        for i, c in enumerate(contains):
            stmt_node = _reconstruct_stmt(c, idx, f"{path}/body/{i}", path_map)
            body.append(stmt_node)
        return body

    # Last resort: docstring only (no stmt_seq, no contains)
    doc_members = idx.get(owner, "docstring")
    if doc_members:
        doc_sym = doc_members[0]
        text_members = idx.get(doc_sym, "text_content")
        text = text_members[0] if text_members else ""
        body.append({"_type": "Expr", "value": {"_type": "Constant", "value": text}})

    return body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_ast_tag_jsonl_to_ast_json(
    records: List[Dict],
) -> Tuple[List[Dict], Dict[str, str]]:
    """Convert AstTagRelation records to AST JSON tree.

    Returns:
        (json_nodes, inner_path_to_tmp_var_name)
        - json_nodes: list of top-level AST JSON statement dicts
        - inner_path_to_tmp_var_name: mapping of jsondiff paths to $tmp_var names
    """
    idx = _RelationIndex(records)
    path_map: Dict[str, str] = {}

    # Top-level: <module> contains top-level statements
    module_contains = idx.get("<module>", "contains")
    if not module_contains:
        return [], path_map

    # Use stmt_seq for ordering if available
    stmt_seq = idx.get("<module>", "stmt_seq")
    if stmt_seq:
        ordered_syms = []
        for seq_entry in stmt_seq:
            parts = seq_entry.split(":", 1)
            ordered_syms.append(parts[1] if len(parts) == 2 else seq_entry)
    else:
        ordered_syms = module_contains

    json_nodes = []
    for i, sym in enumerate(ordered_syms):
        node = _reconstruct_stmt(sym, idx, f"/body/{i}", path_map)
        json_nodes.append(node)

    return json_nodes, path_map


if __name__ == "__main__":
    import os

    # Test: load a JSONL file, convert to AST JSON, print result
    dataset_dir = os.path.join(os.path.dirname(__file__),
                               "..", "example", "tag_auto_encoder", "dataset")
    test_file = os.path.join(dataset_dir, "symbolic_tensor", "tensor_util",
                             "none_tensor_like.jsonl")
    if os.path.exists(test_file):
        records = []
        with open(test_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} records from {os.path.basename(test_file)}")

        json_nodes, path_map = convert_ast_tag_jsonl_to_ast_json(records)
        print(f"Reconstructed {len(json_nodes)} top-level statements")
        print(f"Path map has {len(path_map)} entries")
        print(json.dumps(json_nodes, indent=2, default=str)[:2000])
    else:
        print(f"Test file not found: {test_file}")

    # Roundtrip test: Python -> JSONL -> AST JSON
    print("\n=== Roundtrip test ===")
    from experience.ast_tag.convert_python_to_ast_tag_jsonl import convert_python_to_ast_tag_jsonl, ast_to_dict

    sample_code = '''
def hello(name: str = "world") -> str:
    """Say hello."""
    return f"Hello, {name}!"

x = hello("Alice")
'''
    jsonl_str = convert_python_to_ast_tag_jsonl(sample_code)
    records = [json.loads(line) for line in jsonl_str.strip().splitlines()]
    print(f"Forward: {len(records)} relations")

    json_nodes, path_map = convert_ast_tag_jsonl_to_ast_json(records)
    print(f"Reverse: {len(json_nodes)} top-level statements")
    for node in json_nodes:
        print(f"  {node.get('_type', '?')}: {json.dumps(node, default=str)[:120]}")
    print(f"Path map entries: {len(path_map)}")
    print("convert_ast_tag_jsonl_to_ast_json: OK")
