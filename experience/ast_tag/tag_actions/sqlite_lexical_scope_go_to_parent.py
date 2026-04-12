"""
sqlite_lexical_scope_go_to_parent :=
    CodePosition
    <- $ast_tag_db AstTagDB
    <- $file_id str
    <- $member_tag str
    # inline — SQLite implementation
"""

from experience.ast_tag.tag_actions.code_position import CodePosition
from experience.ast_tag.relation_tag_classification import LEXICAL_RELATION_TAGS
from experience.ast_tag.ast_tag_db import AstTagDB


def sqlite_lexical_scope_go_to_parent(
    ast_tag_db: AstTagDB, file_id: str, member_tag: str
) -> CodePosition:
    """Go to Parent: which scope lexically contains this node?

    SQLite implementation. Returns exactly one CodePosition.
    """
    placeholders = ",".join("?" for _ in LEXICAL_RELATION_TAGS)
    cursor = ast_tag_db._conn.execute(
        f"""
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE file_id = ? AND member_tag = ?
          AND relation_tag IN ({placeholders})
        LIMIT 1
        """,
        (file_id, member_tag, *LEXICAL_RELATION_TAGS),
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(
            f"No lexical parent found for member_tag={member_tag!r} in file_id={file_id!r}"
        )
    return CodePosition(*row)
