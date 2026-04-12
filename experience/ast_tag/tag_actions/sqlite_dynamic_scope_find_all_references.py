"""
sqlite_dynamic_scope_find_all_references :=
    list[$reference_site CodePosition]
    <- $ast_tag_db AstTagDB
    <- $symbol_name str
    # inline — SQLite implementation
"""

from experience.ast_tag.tag_actions.code_position import CodePosition
from experience.ast_tag.relation_tag_classification import DYNAMIC_RELATION_TAGS
from experience.ast_tag.ast_tag_db import AstTagDB


def sqlite_dynamic_scope_find_all_references(
    ast_tag_db: AstTagDB, symbol_name: str
) -> list[CodePosition]:
    """Find All References: who uses/calls/references this symbol?

    SQLite implementation. Searches ALL files for dynamic relations
    where member_tag == symbol_name.
    """
    results: list[CodePosition] = []
    seen: set[tuple[str, int, str]] = set()  # (file_id, line, owner_tag) dedup

    # Direct reference lookup with DYNAMIC_RELATION_TAGS
    placeholders = ",".join("?" for _ in DYNAMIC_RELATION_TAGS)
    cursor = ast_tag_db._conn.execute(
        f"""
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE member_tag = ?
          AND relation_tag IN ({placeholders})
        """,
        (symbol_name, *DYNAMIC_RELATION_TAGS),
    )
    for row in cursor.fetchall():
        key = (row[0], row[1], row[2])
        if key not in seen:
            seen.add(key)
            results.append(CodePosition(*row))

    # Attribute references: Attribute.attr matches symbol_name
    attr_cursor = ast_tag_db._conn.execute(
        """
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE relation_tag = 'Attribute__attr' AND member_tag = ?
        """,
        (symbol_name,),
    )
    for row in attr_cursor.fetchall():
        key = (row[0], row[1], row[2])
        if key not in seen:
            seen.add(key)
            results.append(CodePosition(*row))

    return results
