"""
Tests for LLMAnomalyVerifier.

Covers:
- Module imports without anthropic SDK
- Prompt never contains raw content fields
- AnomalyResult from mock response is valid
- Fallback on JSON parse failure
- Markdown fence stripping
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult, NormalizedIntent
from axor_classifier_llm.verifier import (
    LLMAnomalyVerifier,
    _build_user_message,
    _format_window,
    _parse_response,
)


def _ni(**kw) -> NormalizedIntent:
    defaults = dict(
        tool="read", operation="file_read", target_kind="workdir",
        destination_kind="none", provenance="repo",
        reads_secret_like_data=False, writes_outside_workdir=False,
        executes_generated_code=False, after_external_read=False,
        after_secret_access=False, data_flow="none",
    )
    defaults.update(kw)
    return NormalizedIntent(**defaults)


def _mock_client(response_text: str) -> MagicMock:
    """Returns a mock AsyncAnthropic client that returns the given text."""
    content = MagicMock()
    content.text = response_text
    response = MagicMock()
    response.content = [content]
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


# ── Import safety ───────────────────────────────────────────────────────────────

def test_module_imports_without_anthropic(monkeypatch):
    """Package must import without anthropic SDK; error deferred to construction."""
    import importlib
    import sys
    # Simulate anthropic not installed by blocking the import
    monkeypatch.setitem(sys.modules, "anthropic", None)
    # verifier module itself should import fine (TYPE_CHECKING guard)
    import axor_classifier_llm.verifier as mod
    importlib.reload(mod)  # re-import with anthropic=None in sys.modules


def test_construction_raises_without_anthropic(monkeypatch):
    """ImportError raised at construction time, not import time."""
    import sys
    monkeypatch.setitem(sys.modules, "anthropic", None)
    from axor_classifier_llm.verifier import LLMAnomalyVerifier as V
    with pytest.raises(ImportError, match="anthropic"):
        V(client=MagicMock())


def test_package_init_imports():
    from axor_classifier_llm import LLMAnomalyVerifier
    assert LLMAnomalyVerifier is not None


# ── _format_window ──────────────────────────────────────────────────────────────

def test_format_window_no_raw_content():
    """Formatted window must not contain any raw content strings."""
    window = [
        _ni(tool="read", operation="file_read", target_kind="external_url",
            provenance="external_web"),
        _ni(tool="bash", operation="execute_generated_code",
            executes_generated_code=True, after_external_read=True),
    ]
    output = _format_window(window)
    # Only normalized field values — no raw file contents, no URLs, no tool output
    assert "tool=" in output
    assert "op=" in output
    assert "target=" in output
    # Verify no large blobs or newlines-within-intent format
    for line in output.splitlines():
        assert len(line) < 300, f"line suspiciously long: {line!r}"


def test_format_window_flags_present():
    window = [_ni(reads_secret_like_data=True, after_external_read=True)]
    output = _format_window(window)
    assert "reads_secret" in output
    assert "after_external_read" in output


def test_format_window_no_flags_when_normal():
    window = [_ni()]
    output = _format_window(window)
    assert "[" not in output


def test_format_window_empty():
    assert _format_window([]) == ""


# ── _build_user_message ─────────────────────────────────────────────────────────

def test_user_message_contains_no_raw_content_fields():
    """The user message must reference only NormalizedIntent fields."""
    window = [_ni(target_kind="secret", reads_secret_like_data=True)]
    msg = _build_user_message(window, task_signal_hint="coding", policy_name="strict")
    assert "Task type: coding" in msg
    assert "Active policy: strict" in msg
    assert "Behavioral sequence" in msg
    # Should NOT contain field names that would imply raw content was leaked
    for forbidden in ("raw_input", "tool_result", "content", "webpage", "chain_of_thought"):
        assert forbidden not in msg.lower(), f"found forbidden field: {forbidden!r}"


def test_user_message_without_hints():
    window = [_ni()]
    msg = _build_user_message(window, task_signal_hint="", policy_name="")
    assert "Behavioral sequence" in msg
    assert "Task type" not in msg
    assert "Active policy" not in msg


# ── _parse_response ─────────────────────────────────────────────────────────────

def test_parse_valid_critical_response():
    text = json.dumps({"score": 0.9, "class": "critical", "reasons": ["exfil_chain"]})
    result = _parse_response(text)
    assert result.cls == AnomalyClass.CRITICAL
    assert result.score == pytest.approx(0.9)
    assert "exfil_chain" in result.reasons


def test_parse_valid_suspicious_response():
    text = json.dumps({"score": 0.55, "class": "suspicious", "reasons": ["unexpected_tool"]})
    result = _parse_response(text)
    assert result.cls == AnomalyClass.SUSPICIOUS


def test_parse_valid_normal_response():
    text = json.dumps({"score": 0.1, "class": "normal", "reasons": []})
    result = _parse_response(text)
    assert result.cls == AnomalyClass.NORMAL
    assert result.score == pytest.approx(0.1)


def test_parse_strips_markdown_fence():
    text = '```json\n{"score": 0.8, "class": "critical", "reasons": ["test"]}\n```'
    result = _parse_response(text)
    assert result.cls == AnomalyClass.CRITICAL


def test_parse_fallback_on_invalid_json():
    result = _parse_response("this is not json at all")
    assert result.cls == AnomalyClass.SUSPICIOUS
    assert "verifier_parse_error" in result.reasons
    assert result.score == pytest.approx(0.5)


def test_parse_fallback_on_unknown_class():
    text = json.dumps({"score": 0.5, "class": "unknown_class", "reasons": []})
    result = _parse_response(text)
    assert result.cls == AnomalyClass.SUSPICIOUS
    assert "verifier_parse_error" in result.reasons


# ── LLMAnomalyVerifier.verify ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_returns_anomaly_result():
    response_text = json.dumps({"score": 0.85, "class": "critical", "reasons": ["exfil"]})
    client = _mock_client(response_text)
    verifier = LLMAnomalyVerifier(client=client, model="claude-test", max_tokens=128)
    window = [
        _ni(target_kind="external_url", provenance="external_web"),
        _ni(reads_secret_like_data=True, after_external_read=True),
        _ni(operation="network_request", data_flow="local_to_external",
            after_secret_access=True),
    ]
    result = await verifier.verify(window)
    assert isinstance(result, AnomalyResult)
    assert result.cls == AnomalyClass.CRITICAL
    assert result.score == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_verify_passes_task_signal_hint():
    response_text = json.dumps({"score": 0.1, "class": "normal", "reasons": []})
    client = _mock_client(response_text)
    verifier = LLMAnomalyVerifier(client=client)
    await verifier.verify([_ni()], task_signal_hint="coding", policy_name="default")
    call_kwargs = client.messages.create.call_args.kwargs
    user_msg = call_kwargs["messages"][0]["content"]
    assert "coding" in user_msg
    assert "default" in user_msg


@pytest.mark.asyncio
async def test_verify_prompt_has_no_raw_content():
    """The prompt sent to the LLM must not include any raw tool content."""
    response_text = json.dumps({"score": 0.2, "class": "normal", "reasons": []})
    client = _mock_client(response_text)
    verifier = LLMAnomalyVerifier(client=client)
    window = [_ni(target_kind="secret", reads_secret_like_data=True)]
    await verifier.verify(window, task_signal_hint="research", policy_name="strict")

    call_kwargs = client.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    system_content = call_kwargs.get("system", "")

    for content in (user_content, system_content):
        for forbidden in ("raw_input", "tool_result", "webpage", "chain_of_thought"):
            assert forbidden not in content.lower(), (
                f"forbidden field {forbidden!r} found in LLM prompt"
            )
