from __future__ import annotations

import pytest

from smallctl.tools.ssh_files_patch_utils import best_patch_match as local_best_patch_match
from smallctl.tools import ssh_files_remote_helper as remote_helper_module


def _load_remote_best_patch_match():
    """Load the remote helper source and return its best_patch_match function.

    The helper source ends with a ``main()`` invocation that is meant to run on
    a remote host.  We strip that call so we can import just the function
    definitions into a restricted namespace for testing.
    """
    source = remote_helper_module._REMOTE_HELPER_SOURCE
    stripped = source.rstrip()
    if stripped.endswith("main()"):
        stripped = stripped[: -len("main()")].rstrip()
    namespace = {}
    exec(stripped, namespace)
    return namespace["best_patch_match"]


@pytest.fixture(scope="module")
def remote_best_patch_match():
    return _load_remote_best_patch_match()


@pytest.fixture(scope="module", params=["local", "remote"])
def best_patch_match(request, remote_best_patch_match):
    if request.param == "local":
        return local_best_patch_match
    return remote_best_patch_match


def test_best_patch_match_exact(best_patch_match):
    content = "alpha\nbeta\ngamma\n"
    target = "beta"
    result = best_patch_match(content, target)
    assert result is not None
    assert result["preview"] == "beta"
    assert result["start_line"] == 2
    assert result["end_line"] == 2
    assert result["similarity"] == 1.0
    assert result["match_basis"] == "exact"


def test_best_patch_match_fuzzy(best_patch_match):
    content = "alpha\nbeta   gamma\n"
    target = "alpha\nbeta gamma"
    result = best_patch_match(content, target)
    assert result is not None
    assert result["start_line"] == 1
    assert result["end_line"] == 2
    assert result["match_basis"] == "whitespace_normalized"
    assert result["similarity"] == 1.0


def test_best_patch_match_multiple_candidates(best_patch_match):
    content = "def foo():\n    pass\ndef bar():\n    pass\n"
    target = "def foo():\n    pass"
    result = best_patch_match(content, target)
    assert result is not None
    assert result["start_line"] == 1
    assert result["end_line"] == 2
    assert result["similarity"] == 1.0
    assert result["match_basis"] == "exact"


def test_best_patch_match_no_qualified_match(best_patch_match):
    content = "alpha\nbeta\ngamma\n"
    target = "zzz\nxyz"
    result = best_patch_match(content, target)
    assert result is not None
    assert result["similarity"] < 0.5


def test_best_patch_match_empty_inputs(best_patch_match):
    assert best_patch_match("", "foo") is None
    assert best_patch_match("foo", "") is None
    assert best_patch_match("", "") is None


def test_best_patch_match_local_remote_consistency(remote_best_patch_match):
    cases = [
        ("alpha\nbeta\ngamma\n", "beta"),
        ("alpha\nbeta   gamma\n", "alpha\nbeta gamma"),
        ("def foo():\n    pass\ndef bar():\n    pass\n", "def foo():\n    pass"),
        ("alpha\nbeta\ngamma\n", "zzz\nxyz"),
        ("", "foo"),
        ("foo", ""),
        ("", ""),
        ("only one line", "one"),
        ("line1\nline2\nline3\nline4\n", "line2\nline3"),
    ]
    for content, target in cases:
        local = local_best_patch_match(content, target)
        remote = remote_best_patch_match(content, target)
        assert local == remote, f"divergence for content={content!r}, target={target!r}"
