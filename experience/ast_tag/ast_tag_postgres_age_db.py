"""
AstTagPostgresAgeDB — PostgreSQL + Apache AGE Implementation of AstTagDB

Uses Apache AGE graph extension for openCypher queries.
Each relation_tag maps to an edge label in the graph.

Graph name: ast_tag (default)

Node labels: root, inner, leaf_symbol, leaf_literal
Edge labels: one per relation_tag (e.g. FunctionDef__body, Call__func)
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from experience.ast_tag.ast_tag_db import AstTagDB
from experience.ast_tag.relation_tag_classification import (
    LEXICAL_RELATION_TAGS,
    DYNAMIC_RELATION_TAGS,
)


# ═══════════════════════════════════════════════════════════
# Node label classification
# ═══════════════════════════════════════════════════════════

_LITERAL_KEYWORDS = frozenset({"True", "False", "None", "..."})
_LITERAL_PREFIX_RE = re.compile(
    r"""^(?:
        [+-]?\d+(?:\.\d*)?          # int or float: 42, 3.14, -1
        | [+-]?\.\d+                # float: .5
        | '.*'                      # single-quoted repr
        | ".*"                      # double-quoted repr
        | b'.*'                     # bytes repr
        | b".*"                     # bytes repr
    )$""",
    re.VERBOSE | re.DOTALL,
)


def classify_node_label(symbol: str) -> str:
    """Classify a symbol string into its AGE node label."""
    if symbol == "<module>":
        return "root"
    if symbol.startswith("$"):
        return "inner"
    if symbol in _LITERAL_KEYWORDS:
        return "leaf_literal"
    if _LITERAL_PREFIX_RE.match(symbol):
        return "leaf_literal"
    # Try numeric parse as fallback
    try:
        float(symbol)
        return "leaf_literal"
    except (ValueError, TypeError):
        pass
    return "leaf_symbol"


def _is_file_scoped(label: str) -> bool:
    """root and inner nodes are file-scoped; leaves are global."""
    return label in ("root", "inner")


# ═══════════════════════════════════════════════════════════
# AGE Cypher helpers
# ═══════════════════════════════════════════════════════════

def _age_setup(conn: "psycopg2.extensions.connection") -> None:
    """Load AGE extension and set search_path for a connection."""
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, \"$user\", public")
    conn.commit()


def _cypher_query(
    conn: "psycopg2.extensions.connection",
    graph_name: str,
    cypher: str,
    params: tuple = (),
    columns: list[str] | None = None,
) -> list[tuple]:
    """Execute a Cypher query via ag_catalog.cypher() and return rows as tuples."""
    if columns is None:
        columns = ["v agtype"]
    col_def = ", ".join(columns)
    sql = f"SELECT * FROM cypher('{graph_name}', $${cypher}$$) as t({col_def})"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    except Exception:
        conn.rollback()
        raise


def _cypher_exec(
    conn: "psycopg2.extensions.connection",
    graph_name: str,
    cypher: str,
) -> None:
    """Execute a Cypher statement with no result set."""
    sql = f"SELECT * FROM cypher('{graph_name}', $${cypher}$$) as t(v agtype)"
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def _escape_cypher_string(s: str) -> str:
    """Escape a string for safe embedding in Cypher literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


# ═══════════════════════════════════════════════════════════
# AstTagPostgresAgeDB class
# ═══════════════════════════════════════════════════════════

class AstTagPostgresAgeDB(AstTagDB):
    """PostgreSQL + Apache AGE implementation of AstTagDB interface."""

    def __init__(
        self, conn: "psycopg2.extensions.connection", graph_name: str = "ast_tag"
    ) -> None:
        """
        Wrap an existing psycopg2 connection with AGE extension loaded.
        Assumes graph already exists and is populated.
        Use load_jsonl_dataset_into_ast_tag_age_db() to create a populated instance.
        """
        self._conn = conn
        self._graph_name = graph_name

    @property
    def graph_name(self) -> str:
        return self._graph_name

    # ═══════════════════════════════════════════════════════════
    # AstTagDB interface implementation
    # ═══════════════════════════════════════════════════════════

    def get_all_loaded_file_ids(self) -> list[str]:
        """Enumerate all loaded source files, sorted alphabetically."""
        rows = _cypher_query(
            self._conn,
            self._graph_name,
            "MATCH (r:root) RETURN DISTINCT r.file_id ORDER BY r.file_id",
            columns=["file_id agtype"],
        )
        return [_agtype_to_str(row[0]) for row in rows]

    def count_file_relation_records(self, file_id: str) -> int:
        """Total edges in a file — proxy for file complexity."""
        esc_fid = _escape_cypher_string(file_id)
        rows = _cypher_query(
            self._conn,
            self._graph_name,
            f"MATCH ()-[r]->() WHERE r.file_id = '{esc_fid}' RETURN count(r)",
            columns=["cnt agtype"],
        )
        return _agtype_to_int(rows[0][0]) if rows else 0

    def get_nearby_symbols_around_line_range(
        self,
        file_id: str,
        line_range_start: int,
        line_range_end: int,
        context_margin: int = 5,
    ) -> list[str]:
        """Collect symbols from edges near a line range (excluding the range itself)."""
        esc_fid = _escape_cypher_string(file_id)
        lo = line_range_start - context_margin
        hi = line_range_end + context_margin
        rows = _cypher_query(
            self._conn,
            self._graph_name,
            f"""
            MATCH (owner)-[r]->(member)
            WHERE r.file_id = '{esc_fid}'
              AND r.line >= {lo} AND r.line <= {hi}
              AND NOT (r.line >= {line_range_start} AND r.line <= {line_range_end})
            RETURN DISTINCT owner.symbol, member.symbol
            """,
            columns=["owner_sym agtype", "member_sym agtype"],
        )
        symbols: set[str] = set()
        for owner_sym, member_sym in rows:
            symbols.add(_agtype_to_str(owner_sym))
            symbols.add(_agtype_to_str(member_sym))
        return sorted(symbols)

    def execute_raw_query(
        self, query_statement: str, query_params: tuple = ()
    ) -> list[tuple]:
        """Raw Cypher query escape hatch for ad-hoc analysis."""
        with self._conn.cursor() as cur:
            cur.execute(query_statement, query_params)
            return cur.fetchall()

    # ═══════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════

    def close(self) -> None:
        """Close the underlying psycopg2 connection."""
        self._conn.close()


# ═══════════════════════════════════════════════════════════
# agtype parsing helpers
# ═══════════════════════════════════════════════════════════

def _agtype_to_str(val: Any) -> str:
    """Extract a Python str from an agtype value."""
    if isinstance(val, str):
        # AGE returns agtype as strings like '"value"' — strip quotes
        if val.startswith('"') and val.endswith('"'):
            return val[1:-1]
        return val
    return str(val)


def _agtype_to_int(val: Any) -> int:
    """Extract a Python int from an agtype value."""
    if isinstance(val, int):
        return val
    return int(str(val))


# ═══════════════════════════════════════════════════════════
# Schema management
# ═══════════════════════════════════════════════════════════

def create_ast_tag_age_graph(
    conn: "psycopg2.extensions.connection", graph_name: str = "ast_tag"
) -> None:
    """
    Create the AGE graph and vertex/edge labels.
    Idempotent: checks existence before creating.
    """
    _age_setup(conn)

    # Create graph if not exists
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s",
            (graph_name,),
        )
        if cur.fetchone()[0] == 0:
            cur.execute("SELECT create_graph(%s)", (graph_name,))
    conn.commit()

    # Create vertex labels
    for vlabel in ("root", "inner", "leaf_symbol", "leaf_literal"):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM ag_catalog.ag_label "
                "WHERE name = %s AND graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)",
                (vlabel, graph_name),
            )
            if cur.fetchone()[0] == 0:
                cur.execute(f"SELECT create_vlabel('{graph_name}', '{vlabel}')")
        conn.commit()

    # Create edge labels — one per relation_tag
    all_relation_tags = LEXICAL_RELATION_TAGS | DYNAMIC_RELATION_TAGS
    for elabel in sorted(all_relation_tags):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM ag_catalog.ag_label "
                "WHERE name = %s AND graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)",
                (elabel, graph_name),
            )
            if cur.fetchone()[0] == 0:
                cur.execute(f"SELECT create_elabel('{graph_name}', '{elabel}')")
        conn.commit()


# ═══════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════

def _batch_cypher_exec(
    conn: "psycopg2.extensions.connection",
    graph_name: str,
    cypher: str,
) -> None:
    """Execute a (possibly large) Cypher mutation with no result needed."""
    sql = f"SELECT * FROM cypher('{graph_name}', $${cypher}$$) as t(v agtype)"
    with conn.cursor() as cur:
        cur.execute(sql)


def load_jsonl_dataset_into_ast_tag_age_db(
    dataset_dir: str,
    conn_params: str = "dbname=ast_tag",
    graph_name: str = "ast_tag",
) -> AstTagPostgresAgeDB:
    """
    Walk dataset_dir, load all .jsonl files into the AGE graph.
    Direct SQL bulk-insert into AGE internal tables (bypasses Cypher).
    Returns a ready-to-query AstTagPostgresAgeDB.
    """
    import time as _time
    from psycopg2.extras import execute_values

    t0 = _time.perf_counter()

    conn = psycopg2.connect(conn_params)
    conn.autocommit = False

    # Drop existing graph to avoid duplicate edges on re-load
    _age_setup(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s",
            (graph_name,),
        )
        if cur.fetchone()[0] > 0:
            cur.execute("SELECT drop_graph(%s, true)", (graph_name,))
    conn.commit()
    # Reconnect — drop_graph(cascade) can invalidate the connection
    conn.close()
    conn = psycopg2.connect(conn_params)
    conn.autocommit = False

    create_ast_tag_age_graph(conn, graph_name)
    _age_setup(conn)

    # ── Get label metadata ───────────────────────────────────────
    vlabel_ids: dict[str, int] = {}
    elabel_ids: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.name, l.id, l.kind
            FROM ag_catalog.ag_label l
            WHERE l.graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)
        """, (graph_name,))
        for name, lid, kind in cur.fetchall():
            if kind == 'v':
                vlabel_ids[name] = lid
            elif kind == 'e':
                elabel_ids[name] = lid

    # ── Parse all records, deduplicate nodes ─────────────────────
    dataset_path = Path(dataset_dir)
    records: list[tuple] = []
    unique_nodes: dict[tuple, bool] = {}  # (label, symbol, file_id|None)

    for jsonl_file in sorted(dataset_path.rglob("*.jsonl")):
        file_id = str(jsonl_file.relative_to(dataset_path))
        with open(jsonl_file, "r") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                record = json.loads(raw_line)
                ot = str(record["owner_tag"])
                mt = str(record["member_tag"])
                rt = record["relation_tag"]
                ln = record["line"]
                mov = record["member_order_value"]

                ol = classify_node_label(ot)
                ml = classify_node_label(mt)

                ok = (ol, ot, file_id if _is_file_scoped(ol) else None)
                mk = (ml, mt, file_id if _is_file_scoped(ml) else None)
                unique_nodes[ok] = True
                unique_nodes[mk] = True

                records.append((file_id, ot, ol, mt, ml, rt, ln, mov))

    t_parse = _time.perf_counter()

    # ── Phase 1: Bulk-INSERT unique nodes into AGE vertex tables ─
    node_gids: dict[tuple, int] = {}  # (label, symbol, file_id|None) -> graphid

    for vlabel in ("root", "inner", "leaf_symbol", "leaf_literal"):
        label_id = vlabel_ids.get(vlabel)
        if label_id is None:
            continue
        nodes = [(sym, fid) for (l, sym, fid) in unique_nodes if l == vlabel]
        if not nodes:
            continue

        seq = f'"{graph_name}"."{vlabel}_id_seq"'

        # Build (props_json,) tuples for execute_values
        props_data: list[tuple[str]] = []
        node_keys: list[tuple] = []
        for sym, fid in nodes:
            if fid is not None:
                props_data.append((json.dumps({"symbol": sym, "file_id": fid}),))
            else:
                props_data.append((json.dumps({"symbol": sym}),))
            node_keys.append((vlabel, sym, fid))

        with conn.cursor() as cur:
            result = execute_values(
                cur,
                f'INSERT INTO "{graph_name}"."{vlabel}" (id, properties)'
                f' VALUES %s RETURNING id',
                props_data,
                template=f"(ag_catalog._graphid({label_id},"
                         f" nextval('{seq}')), %s::agtype)",
                fetch=True,
                page_size=10000,
            )

        # Map keys to graphids (returned in insertion order)
        for idx, (gid,) in enumerate(result):
            node_gids[node_keys[idx]] = gid

    conn.commit()
    t_nodes = _time.perf_counter()

    # ── Phase 2: Bulk-INSERT edges into AGE edge tables ──────────
    # Group edges by relation_tag (= edge label)
    edges_by_label: dict[str, list[tuple]] = {}
    for fid, ot, ol, mt, ml, rt, ln, mov in records:
        ok = (ol, ot, fid if _is_file_scoped(ol) else None)
        mk = (ml, mt, fid if _is_file_scoped(ml) else None)
        start_gid = node_gids[ok]
        end_gid = node_gids[mk]
        props = json.dumps({
            "file_id": fid, "line": ln, "member_order_value": mov,
        })
        edges_by_label.setdefault(rt, []).append((start_gid, end_gid, props))

    for elabel, edge_data in edges_by_label.items():
        label_id = elabel_ids.get(elabel)
        if label_id is None:
            continue
        seq = f'"{graph_name}"."{elabel}_id_seq"'

        with conn.cursor() as cur:
            execute_values(
                cur,
                f'INSERT INTO "{graph_name}"."{elabel}"'
                f' (id, start_id, end_id, properties) VALUES %s',
                edge_data,
                template=f"(ag_catalog._graphid({label_id},"
                         f" nextval('{seq}')),"
                         f" %s::graphid, %s::graphid, %s::agtype)",
                page_size=10000,
            )

    conn.commit()
    t_edges = _time.perf_counter()
    total = t_edges - t0
    print(
        f"Loaded {len(records)} records into graph '{graph_name}' "
        f"from {dataset_dir}\n"
        f"  parse: {t_parse - t0:.2f}s | "
        f"nodes ({len(unique_nodes)}): {t_nodes - t_parse:.2f}s | "
        f"edges ({len(records)}): {t_edges - t_nodes:.2f}s | "
        f"total: {total:.2f}s"
    )
    return AstTagPostgresAgeDB(conn, graph_name)


if __name__ == "__main__":
    import sys

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "test_dataset"
    )
    conn_params = sys.argv[2] if len(sys.argv) > 2 else "dbname=ast_tag"
    db = load_jsonl_dataset_into_ast_tag_age_db(dataset_dir, conn_params)
    file_ids = db.get_all_loaded_file_ids()
    print(f"\n{len(file_ids)} files loaded:")
    for fid in file_ids:
        count = db.count_file_relation_records(fid)
        print(f"  {fid}: {count} records")
    total = sum(db.count_file_relation_records(fid) for fid in file_ids)
    print(f"\nTotal: {total} records across {len(file_ids)} files")
    print("ast_tag_postgres_age_db: OK")
