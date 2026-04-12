"""
Benchmark: query-only timing for SQLite vs PostgreSQL AGE backends.
Loads each DB once, then times 100 calls per tag_action.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tag_actions.dynamic_scope_find_all_references import dynamic_scope_find_all_references
from tag_actions.dynamic_scope_go_to_definition import dynamic_scope_go_to_definition
from tag_actions.lexical_scope_expand_children import lexical_scope_expand_children
from tag_actions.lexical_scope_go_to_parent import lexical_scope_go_to_parent

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "test_dataset")

# ── Collect test inputs from SQLite (fast) ──────────────────────────

from ast_tag_db import load_jsonl_dataset_into_ast_tag_db

sqlite_db = load_jsonl_dataset_into_ast_tag_db(DATASET_DIR)
file_ids = sqlite_db.get_all_loaded_file_ids()

# Gather symbol names for dynamic scope tests
_sym_rows = sqlite_db._conn.execute(
    "SELECT DISTINCT member_tag FROM relations "
    "WHERE relation_tag = 'FunctionDef__name' LIMIT 100"
).fetchall()
symbol_names = [r[0] for r in _sym_rows]

# Gather (file_id, owner_tag) pairs for lexical expand_children
_owner_rows = sqlite_db._conn.execute(
    "SELECT DISTINCT file_id, owner_tag FROM relations LIMIT 100"
).fetchall()
owner_pairs = [(r[0], r[1]) for r in _owner_rows]

# Gather (file_id, member_tag) pairs for lexical go_to_parent
_member_rows = sqlite_db._conn.execute(
    "SELECT DISTINCT file_id, member_tag FROM relations LIMIT 100"
).fetchall()
member_pairs = [(r[0], r[1]) for r in _member_rows]


def bench(label, fn, db, inputs):
    """Time fn(db, *args) over all inputs, return total seconds."""
    start = time.perf_counter()
    for args in inputs:
        if not isinstance(args, tuple):
            args = (args,)
        try:
            fn(db, *args)
        except (ValueError, NotImplementedError):
            pass
    elapsed = time.perf_counter() - start
    return elapsed


def run_suite(tag, db):
    results = {}
    n = len(symbol_names)

    t = bench("find_all_references", dynamic_scope_find_all_references, db,
              symbol_names)
    results["find_all_references"] = (n, t)

    t = bench("go_to_definition", dynamic_scope_go_to_definition, db,
              symbol_names)
    results["go_to_definition"] = (n, t)

    n2 = len(owner_pairs)
    t = bench("expand_children", lexical_scope_expand_children, db,
              owner_pairs)
    results["expand_children"] = (n2, t)

    n3 = len(member_pairs)
    t = bench("go_to_parent", lexical_scope_go_to_parent, db,
              member_pairs)
    results["go_to_parent"] = (n3, t)

    return results


# ── SQLite ──────────────────────────────────────────────────────────

print("=== SQLite (query only) ===")
sqlite_results = run_suite("sqlite", sqlite_db)
for name, (n, t) in sqlite_results.items():
    print(f"  {name}: {n} calls in {t:.4f}s ({t/n*1000:.2f} ms/call)")

# ── PostgreSQL AGE ──────────────────────────────────────────────────

pg_db = None
try:
    conn_params = os.environ.get("AST_TAG_PG_CONN", "dbname=ast_tag")
    graph_name = os.environ.get("AST_TAG_PG_GRAPH", "ast_tag")
    from ast_tag_postgres_age_db import load_jsonl_dataset_into_ast_tag_age_db
    print("\nLoading PostgreSQL AGE graph (one-time)...")
    load_start = time.perf_counter()
    pg_db = load_jsonl_dataset_into_ast_tag_age_db(DATASET_DIR, conn_params, graph_name)
    load_time = time.perf_counter() - load_start
    print(f"  Load time: {load_time:.2f}s")
except ImportError:
    print("\npsycopg2 not installed — skipping PostgreSQL AGE benchmark")

if pg_db:
    print("\n=== PostgreSQL AGE (query only) ===")
    pg_results = run_suite("postgres_age", pg_db)
    for name, (n, t) in pg_results.items():
        print(f"  {name}: {n} calls in {t:.4f}s ({t/n*1000:.2f} ms/call)")

    # ── Summary ─────────────────────────────────────────────────────

    print("\n=== Comparison ===")
    print(f"{'action':<30} {'SQLite':>10} {'AGE':>10} {'ratio':>8}")
    print("-" * 62)
    for name in sqlite_results:
        sn, st = sqlite_results[name]
        pn, pt = pg_results[name]
        ratio = pt / st if st > 0 else float("inf")
        print(f"{name:<30} {st:>9.4f}s {pt:>9.4f}s {ratio:>7.1f}x")
    s_total = sum(t for _, t in sqlite_results.values())
    p_total = sum(t for _, t in pg_results.values())
    ratio_total = p_total / s_total if s_total > 0 else float("inf")
    print("-" * 62)
    print(f"{'TOTAL':<30} {s_total:>9.4f}s {p_total:>9.4f}s {ratio_total:>7.1f}x")
