from __future__ import annotations

CHUNK_WRITE_LOOP_GUARD_TOOLS = {"file_write", "file_append"}
WRITE_TOOLS = {"file_write", "file_append", "file_patch", "ast_patch"}

TERMINAL_WRITE_SESSION_REPAIR_KEY = "_terminal_write_session_repair_signatures"
MISSING_FIRST_WRITE_SESSION_RECOVERY_KEY = "_missing_first_write_session_recovery_signatures"
