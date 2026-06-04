from __future__ import annotations

import re

_CONTINUE_DIRECTIVE_RE = re.compile(
    r"^\s*(?:continue|cntinue|conitnue|continune|cotinue|keep\s+going|resume|proceed|go\s+on|carry\s+on)\b"
    r"(?P<suffix>\s*[,;:.-]\s*|\s+(?:and|by|with|to|then|next)\s+).+",
    re.IGNORECASE | re.DOTALL,
)
_CONTINUATION_ACTION_LEAD_RE = re.compile(
    r"^\s*(?:yes|yep|yeah|ok|okay|sure|go\s+ahead|do\s+it|please\s+do|start|begin|run|do|use|try)\b",
    re.IGNORECASE,
)
_WEB_RESEARCH_DIRECTIVE_RE = re.compile(
    r"\b(?:"
    r"web\s*search|websearch|search\s+(?:the\s+)?web|internet\s+search|"
    r"web\s+lookup|look\s+(?:it\s+)?up|browse|research"
    r")\b",
    re.IGNORECASE,
)
_RESEARCH_CONTEXT_RE = re.compile(
    r"\b(?:research|web|internet|online|search|look\s+up|lookup|summary|summarize|summarise|main\s+plot|plot\s+points)\b",
    re.IGNORECASE,
)
_INLINE_USER_WRAP_MARKER_RE = re.compile(
    r"\.\s*User\s+(?P<kind>follow-up|correction):\s*",
    re.IGNORECASE,
)
_FOLLOWUP_FILLERS = {"please", "pls", "now", "again", "just", "then", "more", "further"}
_NUMBERED_OPTION_RE = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$")
_INLINE_NUMBERED_OPTION_RE = re.compile(r"(?:^|\s)(\d+)[.)]\s+(.+?)(?=(?:\s+\d+[.)]\s+)|$)")
_MARKDOWN_OPTION_RE = re.compile(
    r"^\s*(?:\*\*)?(?:option|proposal)\s+(\d+)\s*(?:[-—–:]|\*\*)\s*(.*?)(?:\*\*)?\s*$",
    re.IGNORECASE,
)
_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_OPTION_ACTION_WORDS = re.compile(
    r"\b(stream|streaming|md5|hash|patch|edit|modify|fix|update|implement|add|replace|refactor|test|skip|handle|read|write|calculate)\b",
    re.IGNORECASE,
)
_TARGET_NEGATION_RE = re.compile(r"\b(?:instead\s+of|rather\s+than|without|avoid|do\s+not|don't|dont)\b", re.IGNORECASE)
_TARGET_REPLACEMENT_RE = re.compile(
    r"\b(?:in|as|using|with)\s+(?:rust|go|typescript|javascript|python|bash|shell)\b",
    re.IGNORECASE,
)
_TARGET_LANGUAGE_RE = re.compile(r"\b(?:rust|go|typescript|javascript|python|bash|shell)\b", re.IGNORECASE)
_ORDINAL_FOLLOWUP_RE = re.compile(
    r"\b(?:start\s+(?:with|by|on)|do|use|choose|pick|implement|patch|apply)\s+"
    r"(?:option\s+|proposal\s+|#)?(\d+)\b"
    r"|"
    r"\b(?:option|proposal)\s+#?(\d+)\b",
    re.IGNORECASE,
)
_ORDINAL_WORD_FOLLOWUP_RE = re.compile(
    r"^\s*(?:start\s+(?:with|by|on)|do|use|choose|pick|implement|patch|apply)\s+"
    r"(?:the\s+)?(?P<word>first|second|third|fourth|fifth)\s+(?:one|option|proposal)?\b"
    r"|"
    r"^\s*(?:option|proposal)\s+(?P<option_word>first|second|third|fourth|fifth)\b",
    re.IGNORECASE,
)
_ORDINAL_PREFIX_RE = re.compile(
    r"^\s*(?:start\s+(?:with|by|on)|do|use|choose|pick|implement|patch|apply)\s+"
    r"(?:option\s+|proposal\s+|#)?\d+[.)]?\s*[,;:]?\s*",
    re.IGNORECASE,
)
_SEQUENTIAL_REMOTE_FOLLOWUP_RE = re.compile(
    r"\b(?:now|next|then|proceed|continue|move)\s+(?:to|on|with|do|edit|modify|fix|update|implement|patch|write)\b|"
    r"\b(?:do|edit|modify|fix|update|implement|patch|write)\s+(?:next|now|then)\b",
    re.IGNORECASE,
)
_GENERIC_EDIT_LEAD_RE = re.compile(
    r"^\s*(?:patch|edit|modify|fix|update|implement|apply)\b[^,.;:]*[,;:]\s*",
    re.IGNORECASE,
)
_GENERIC_TARGET_RE = re.compile(r"\b(?:script|file|module|code|python\s+file)\b", re.IGNORECASE)
_FOLLOWUP_ACTION_RE = re.compile(
    r"\b(?:add|apply|change|choose|decide|decision|edit|fix|implement|make|modify|patch|replace|resolve|update|write)\b",
    re.IGNORECASE,
)
_CONTEXTUAL_REFERENCE_RE = re.compile(
    r"\b(?:"
    r"this|that|it|same\s+(?:file|script|module|code)|"
    r"the\s+(?:file|script|module|code|change|fix|patch)|"
    r"loop(?:ing)?|stuck|repetitive|repeat(?:ing|ed)?|"
    r"you(?:'ve| have)\s+read\s+(?:the\s+)?(?:file\s+)?enough|"
    r"read\s+(?:the\s+)?(?:file\s+)?enough"
    r")\b",
    re.IGNORECASE,
)
_QUALITY_FOLLOWUP_RE = re.compile(
    r"\b(?:"
    r"still|inconsistent|inconsistency|wrong|off|broken|"
    r"not\s+(?:fixed|right|consistent|working)|"
    r"does(?:n't| not)\s+(?:look|match|work)|"
    r"mismatch(?:ed)?|regress(?:ed|ion)?"
    r")\b",
    re.IGNORECASE,
)
_QUALITY_TARGET_RE = re.compile(
    r"\b(?:css|code|file|files|layout|module|page|pages|script|site|style|styles|theme|theming|ui)\b",
    re.IGNORECASE,
)
_GUARD_RECOVERY_NUDGE_RE = re.compile(
    r"\b(?:loop(?:ing)?|stuck|repetitive|repeat(?:ing|ed)?|decide|decision|choose|resolve)\b",
    re.IGNORECASE,
)
_GUARD_FAILURE_RE = re.compile(
    r"\b(?:guard\s+tripped|loop\s+detected|repeated\s+tool|stagnation|stuck\s+in\s+(?:a\s+)?loop|max_consecutive_errors)\b",
    re.IGNORECASE,
)
_RETRY_FOLLOWUP_RE = re.compile(
    r"\b(?:try\s+again|retry|rerun|run\s+it\s+again|attempt\s+again|give\s+it\s+another\s+try)\b",
    re.IGNORECASE,
)
_CORRECTIVE_TOOL_NAMES = (
    "file_patch",
    "ast_patch",
    "file_write",
    "file_append",
    "shell_exec",
    "ssh_exec",
    "task_complete",
)
_CORRECTIVE_RESTEER_RE = re.compile(
    r"\b(?:use|try|call|prefer|switch\s+to|move\s+to)\s+`?"
    r"(?:file_patch|ast_patch|file_write|file_append|shell_exec|ssh_exec|task_complete)"
    r"`?\b(?:\s+(?:instead|now|next))?"
    r"|"
    r"\b(?:not|don't|dont)\s+`?"
    r"(?:file_patch|ast_patch|file_write|file_append|shell_exec|ssh_exec|task_complete)"
    r"`?\b.*\b(?:use|try|call|prefer|switch\s+to|move\s+to)\s+`?"
    r"(?:file_patch|ast_patch|file_write|file_append|shell_exec|ssh_exec|task_complete)"
    r"`?\b",
    re.IGNORECASE,
)
_TASK_BOUNDARY_GUARD_SCRATCHPAD_KEYS = (
    "_tool_attempt_history",
    "_repeat_guard_one_shot_fingerprints",
    "_artifact_read_recovery_nudged",
    "_artifact_read_recovery_query",
    "_artifact_read_synthesis_nudged",
    "_artifact_summary_exit_nudged",
    "_artifact_evidence_unavailable_nudged",
    "_file_read_recovery_nudged",
    "_plan_artifact_read_suppressed",
    "_chunk_write_loop_guard",
    "_chunk_write_loop_guard_config",
    "_chunk_write_loop_guard_read_scheduled",
    "_durable_autocontinue_recoveries",
)
_ACTION_CONFIRMATION_PROMPTS = (
    "would you like me to",
    "do you want me to",
    "should i",
    "shall i",
    "want me to",
    "ready for me to",
)
_AFFIRMATIVE_REMOTE_CONTINUATION_TEXT = "proceed with the approved remote execution steps now"
_IPV4_HOST_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_USER_AT_HOST_RE = re.compile(
    r"\b(?P<user>[A-Za-z0-9._-]+)@(?P<host>[A-Za-z0-9.-]+|\d{1,3}(?:\.\d{1,3}){3})\b",
    re.IGNORECASE,
)
_REMOTE_OPERATIONAL_VERBS = (
    "boot",
    "bring up",
    "create",
    "install",
    "configure",
    "deploy",
    "provision",
    "restart",
    "start",
    "stop",
    "enable",
    "disable",
    "launch",
    "pull",
    "push",
    "run",
    "spin",
    "spin up",
    "switch",
    "switch to",
    "try",
    "upgrade",
    "update",
    "use",
    "setup",
    "set up",
)
_REMOTE_OPERATIONAL_TARGETS = (
    "app",
    "application",
    "container",
    "docker",
    "compose",
    "deployment",
    "image",
    "service",
    "stack",
    "daemon",
    "systemd",
    "startup",
    "server",
    "host",
    "remote",
    "package",
    "packages",
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "apk",
    "pacman",
    "brew",
    "nginx",
    "apache",
    "postgres",
    "postgresql",
    "mysql",
    "mariadb",
    "redis",
    "tracker",
    "vikunja",
)
_REMOTE_DEPLOYMENT_CONTEXT_TARGETS = (
    "container",
    "docker",
    "compose",
    "image",
    "service",
    "deployment",
    "app",
    "application",
    "package",
    "stack",
    "tracker",
)
_REMOTE_CLARIFICATION_PHRASES = (
    "does not have to be",
    "doesn't have to be",
    "does not need to be",
    "doesn't need to be",
    "do not have to be",
    "dont have to be",
    "not have to be",
    "not need to be",
    "will do",
    "any app",
    "any application",
    "any image",
    "any container",
    "any service",
    "exactly",
    "exact name",
    "exact image",
    "called exactly",
)
_REMOTE_LIVE_CORRECTION_PHRASES = (
    "actually use ssh",
    "check again",
    "check it again",
    "do it live",
    "redo the remote action",
    "do not rely on past records",
    "don't rely on past records",
    "dont rely on past records",
    "do not rely on prior records",
    "don't rely on prior records",
    "re-run on the host",
    "rerun on the host",
    "run it live",
    "redo it live",
    "fresh ssh",
    "fresh run",
    "verify again",
)
_REMOTE_LIVE_CORRECTION_HINTS = (
    "actually",
    "again",
    "fresh",
    "live",
    "redo",
    "rerun",
    "re-run",
    "retry",
    "retest",
    "verify",
)
_REMOTE_DIAGNOSTIC_TARGETS = (
    "404",
    "500",
    "502",
    "503",
    "apache",
    "config",
    "configuration",
    "document root",
    "docroot",
    "error",
    "htaccess",
    "live",
    "nginx",
    "not live",
    "page",
    "pages",
    "rewrite",
    "route",
    "routing",
    "serve",
    "serving",
    "server block",
    "site",
    "site structure",
    "vhost",
)
_REMOTE_DIAGNOSTIC_HINTS = (
    "404 error",
    "500 error",
    "502 error",
    "503 error",
    "been update",
    "been updated",
    "does have",
    "error",
    "installed",
    "is live",
    "missing",
    "not live",
    "not serving",
    "serving",
    "updated",
)
_REMOTE_DIAGNOSTIC_QUESTION_RE = re.compile(
    r"^(?:has|have|is|are|does|do|did|why|what|which|where|when|can|could|would)\b",
    re.IGNORECASE,
)
_REMOTE_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w/])/(?:(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+(?:\.[A-Za-z0-9._-]+)?)"
)
_REMOTE_RESIDUE_MARKERS = (
    "htmleof",
    "wrote `htmleof`",
    "wc -l <",
    "written $(",
    "here-doc",
    "heredoc",
    "<< '",
    "<< \"",
)
_REMOTE_CORRECTIVE_CLEANUP_PHRASES = (
    "please fix",
    "fix this",
    "remove this",
    "clean this up",
    "clean it up",
    "trim this",
    "remove this from the bottom of the page",
    "remove this from the end of the page",
    "very bottom of the page",
    "very end of the page",
    "bottom of the page",
    "end of the page",
    "stuck to the very bottom",
    "stuck to the bottom",
    "trailing text",
    "trailing shell echo",
)
_REMOTE_SITE_MUTATION_ACTION_RE = re.compile(
    r"\b(?:"
    r"add|adjust|change|delete|drop|fix|make|modify|remove|replace|restyle|strip|"
    r"swap|turn|update"
    r")\b",
    re.IGNORECASE,
)
_REMOTE_SITE_MUTATION_TARGET_RE = re.compile(
    r"\b(?:"
    r"animation|animations|brand|branded|branding|button|buttons|card|cards|"
    r"color|colors|copy|css|design|font|fonts|footer|header|html|layout|"
    r"logo|logos|page|pages|palette|site|style|styles|styling|text|theme|"
    r"theming|ui|website"
    r")\b",
    re.IGNORECASE,
)
_REMOTE_PERMISSION_FOLLOWUP_RE = re.compile(
    r"\b(?:perm|perms|permission|chmod|owner|mode|right\s+perms)\b",
    re.IGNORECASE,
)
_REMOTE_SCRIPT_HINT_RE = re.compile(
    r"\b(?:script|js|javascript|jave\s+script|java\s+script)\b",
    re.IGNORECASE,
)
_SEMANTIC_RECENT_TAIL_TOKEN_CAP = 320
