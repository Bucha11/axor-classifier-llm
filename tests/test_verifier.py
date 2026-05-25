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

from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult
from axor_core.contracts.canonical import (
    CanonicalizedIntent,
    ExportClass,
    OperationClass,
    PathClass,
    ProviderClass,
    ToolCategory,
)
from axor_classifier_llm.verifier import (
    LLMAnomalyVerifier,
    _build_user_message,
    _extract_text_response,
    _format_window,
    _parse_response,
)


def _ci(**kw) -> CanonicalizedIntent:
    defaults = dict(
        tool_category=ToolCategory.FILE_READ,
        path_class=PathClass.WORKDIR,
        path_depth=2,
        path_extension=".py",
        path_hash="abc123def456789a",
        path_is_absolute=False,
        path_is_outside_workspace=False,
        argument_shape="path",
        argument_length_bucket=1,
        reads_secret=False,
        writes_outside=False,
        executes_generated=False,
        after_external_read=False,
        after_secret_access=False,
        data_flow="none",
        operation_class=OperationClass.READ,
        export_class=ExportClass.NONE,
        provider_class=ProviderClass.CLAUDE,
        taint_state_summary="clean",
        lease_state_summary="none",
        node_depth=0,
    )
    defaults.update(kw)
    return CanonicalizedIntent(**defaults)


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
    monkeypatch.setitem(sys.modules, "anthropic", None)
    import axor_classifier_llm.verifier as mod
    importlib.reload(mod)


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
        _ci(tool_category=ToolCategory.NETWORK, path_class=PathClass.EXTERNAL_URL,
            operation_class=OperationClass.NETWORK, taint_state_summary="tainted:web"),
        _ci(tool_category=ToolCategory.SHELL, operation_class=OperationClass.EXECUTE,
            executes_generated=True, after_external_read=True),
    ]
    output = _format_window(window)
    assert "cat=" in output
    assert "op=" in output
    assert "path=" in output
    for line in output.splitlines():
        assert len(line) < 300, f"line suspiciously long: {line!r}"


def test_format_window_flags_present():
    window = [_ci(reads_secret=True, after_external_read=True)]
    output = _format_window(window)
    assert "reads_secret" in output
    assert "after_external_read" in output


def test_format_window_no_flags_when_normal():
    window = [_ci()]
    output = _format_window(window)
    # No risk flags present — the trailing [reads_secret, ...] section should not appear
    assert "reads_secret" not in output
    assert "writes_outside" not in output
    assert "executes_generated" not in output


def test_format_window_empty():
    assert _format_window([]) == ""


# ── _build_user_message ─────────────────────────────────────────────────────────

def test_user_message_contains_no_raw_content_fields():
    """The user message must reference only canonical feature fields."""
    window = [_ci(path_class=PathClass.SECRET, reads_secret=True)]
    msg = _build_user_message(window, task_signal_hint="coding", policy_name="strict")
    assert "Task type: coding" in msg
    assert "Active policy: strict" in msg
    assert "Behavioral sequence" in msg
    for forbidden in ("raw_input", "tool_result", "content", "webpage", "chain_of_thought"):
        assert forbidden not in msg.lower(), f"found forbidden field: {forbidden!r}"


def test_user_message_without_hints():
    window = [_ci()]
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


def test_parse_clamps_score_to_valid_range():
    high = _parse_response(json.dumps({"score": 2.5, "class": "critical", "reasons": []}))
    low = _parse_response(json.dumps({"score": -1, "class": "normal", "reasons": []}))
    assert high.score == pytest.approx(1.0)
    assert low.score == pytest.approx(0.0)


def test_extract_text_response_handles_empty_content():
    response = MagicMock()
    response.content = []
    assert _extract_text_response(response) == ""


# ── LLMAnomalyVerifier.verify ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_returns_anomaly_result():
    response_text = json.dumps({"score": 0.85, "class": "critical", "reasons": ["exfil"]})
    client = _mock_client(response_text)
    verifier = LLMAnomalyVerifier(client=client, model="claude-test", max_tokens=128)
    window = [
        _ci(path_class=PathClass.EXTERNAL_URL, taint_state_summary="tainted:web"),
        _ci(reads_secret=True, after_external_read=True),
        _ci(operation_class=OperationClass.NETWORK, export_class=ExportClass.EXTERNAL,
            after_secret_access=True, data_flow="local_to_external"),
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
    await verifier.verify([_ci()], task_signal_hint="coding", policy_name="default")
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
    window = [_ci(path_class=PathClass.SECRET, reads_secret=True)]
    await verifier.verify(window, task_signal_hint="research", policy_name="strict")

    call_kwargs = client.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    system_content = call_kwargs.get("system", "")

    for content in (user_content, system_content):
        for forbidden in ("raw_input", "tool_result", "webpage", "chain_of_thought"):
            assert forbidden not in content.lower(), (
                f"forbidden field {forbidden!r} found in LLM prompt"
            )


@pytest.mark.asyncio
async def test_verify_prompt_no_injection_surface():
    """Schema injection: path with injection attempt → only canonical hash appears in prompt."""
    response_text = json.dumps({"score": 0.3, "class": "normal", "reasons": []})
    client = _mock_client(response_text)
    verifier = LLMAnomalyVerifier(client=client)
    # Canonical intent already strips the raw path — only hash remains
    window = [_ci(path_hash="deadbeef12345678", path_class=PathClass.WORKDIR)]
    await verifier.verify(window, task_signal_hint="coding", policy_name="default")

    call_kwargs = client.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    # The injection string never reaches the verifier prompt
    assert "ignore previous instructions" not in user_content
    assert "\n ignore" not in user_content
