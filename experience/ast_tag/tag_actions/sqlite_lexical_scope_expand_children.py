"""
sqlite_lexical_scope_expand_children :=
    list[$child CodePosition]
    <- $ast_tag_db AstTagDB
    <- $file_id str
    <- $owner_tag str
    # inline — SQLite implementation
"""

from experience.ast_tag.tag_actions.code_position import CodePosition
from experience.ast_tag.relation_tag_classification import LEXICAL_RELATION_TAGS
from experience.ast_tag.ast_tag_db import AstTagDB


def sqlite_lexical_scope_expand_children(
    ast_tag_db: AstTagDB, file_id: str, owner_tag: str
) -> list[CodePosition]:
    """Expand Children: what symbols are lexically contained inside this node?

    SQLite implementation.
    """
    placeholders = ",".join("?" for _ in LEXICAL_RELATION_TAGS)
    cursor = ast_tag_db._conn.execute(
        f"""
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE file_id = ? AND owner_tag = ?
          AND relation_tag IN ({placeholders})
        ORDER BY member_order_value
        """,
        (file_id, owner_tag, *LEXICAL_RELATION_TAGS),
    )
    return [CodePosition(*row) for row in cursor.fetchall()]
