from __future__ import annotations

from smallctl.ui.display import check_duplicate_promotion


def test_exact_match_is_duplicate() -> None:
    assert check_duplicate_promotion("Hello world", "Hello world") is True


def test_substring_match_is_duplicate() -> None:
    assert check_duplicate_promotion("Hello world", "Say Hello world today") is True


def test_high_similarity_is_duplicate() -> None:
    assert (
        check_duplicate_promotion(
            "The build passed successfully.",
            "The build passed successfully!",
        )
        is True
    )


def test_greeting_smalltalk_is_duplicate() -> None:
    """Regression for ffb62966: semantically equivalent greetings should be deduped."""
    assert (
        check_duplicate_promotion(
            "Hello! I'm ready to help with whatever you need.",
            "Hello! How can I help you today?",
        )
        is True
    )


def test_different_content_is_not_duplicate() -> None:
    assert (
        check_duplicate_promotion(
            "The cat sat on the mat.",
            "The stock market crashed today.",
        )
        is False
    )


def test_empty_active_text_is_not_duplicate() -> None:
    assert check_duplicate_promotion("Hello!", "") is False


def test_related_but_different_response_is_not_duplicate() -> None:
    """Promotion that adds new information should still be shown."""
    assert (
        check_duplicate_promotion(
            "Build failed: install deps first.",
            "The build failed due to missing dependencies.",
        )
        is False
    )
