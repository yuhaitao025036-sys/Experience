"""
sqlite_dynamic_scope_go_to_definition :=
    list[$definition_site CodePosition]
    <- $ast_tag_db AstTagDB
    <- $symbol_name str
    # inline — SQLite implementation
"""

from experience.ast_tag.tag_actions.code_position import CodePosition
from experience.ast_tag.ast_tag_db import AstTagDB


DEFINITION_RELATION_TAGS = frozenset({
    "FunctionDef__name", "AsyncFunctionDef__name", "ClassDef__name",
    "alias__name",  # imported symbols
})


def sqlite_dynamic_scope_go_to_definition(
    ast_tag_db: AstTagDB, symbol_name: str
) -> list[CodePosition]:
    """Go to Definition: where is this symbol defined?

    SQLite implementation. Searches ALL files for definition relations.
    """
    results: list[CodePosition] = []
    seen: set[tuple[str, str]] = set()  # (file_id, owner_tag) dedup

    # Direct definition lookup across all files
    placeholders = ",".join("?" for _ in DEFINITION_RELATION_TAGS)
    cursor = ast_tag_db._conn.execute(
        f"""
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE relation_tag IN ({placeholders}) AND member_tag = ?
        """,
        (*DEFINITION_RELATION_TAGS, symbol_name),
    )
    for row in cursor.fetchall():
        key = (row[0], row[2])  # (file_id, owner_tag)
        if key not in seen:
            seen.add(key)
            results.append(CodePosition(*row))

    return results
