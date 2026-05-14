# axor-classifier-llm

[![PyPI](https://img.shields.io/pypi/v/axor-classifier-llm)](https://pypi.org/project/axor-classifier-llm/)
[![Python](https://img.shields.io/pypi/pyversions/axor-classifier-llm)](https://pypi.org/project/axor-classifier-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LLM-based gray-zone verifier for axor-core anomaly detection.**

`LLMAnomalyVerifier` uses Anthropic Claude to evaluate ambiguous behavioral sequences that a statistical model cannot confidently resolve.

Zero required dependencies — the Anthropic SDK is an optional extra.

---

## What it does

`MLAnomalyDetector` (from `axor-classifier-simple`) scores `NormalizedIntent` windows and returns a risk class. When a score falls in the **gray zone** — suspicious but not definitively critical — the detector can delegate to an `LLMVerifier` for a second opinion.

`LLMAnomalyVerifier` implements that protocol: it formats `NormalizedIntent` fields into a structured prompt, calls Claude, parses the JSON response, and returns an `AnomalyResult`.

### Security isolation

The verifier receives **only `NormalizedIntent` fields** — behavioral abstractions describing what the agent *tried to do*, not what it saw or produced. Raw tool outputs, webpage content, file contents, and chain-of-thought never enter the prompt.

This design isolates the verifier from content-level prompt injection: a malicious webpage cannot influence the safety verdict by embedding instructions in its text.

---

## Installation

```bash
pip install axor-classifier-llm[llm]
```

The `[llm]` extra installs `anthropic>=0.25`. Without it, the package installs with no dependencies but raises `ImportError` on instantiation with an actionable message.

---

## Quick start

```python
import anthropic
from axor_classifier_llm import LLMAnomalyVerifier

verifier = LLMAnomalyVerifier(
    client=anthropic.AsyncAnthropic(),   # must be async
    model="claude-haiku-4-5-20251001",   # default — fast and cheap
    max_tokens=256,
)

result = await verifier.verify(
    window=normalized_intents,           # list[NormalizedIntent]
    task_signal_hint="focused_mutative", # optional: helps Claude interpret context
    policy_name="focused_mutative",      # optional: helps Claude interpret context
)

print(result.score)    # float 0.0 – 1.0
print(result.cls)      # AnomalyClass.NORMAL | SUSPICIOUS | CRITICAL
print(result.reasons)  # ("secret_access_after_external_read", ...)
```

---

## Integration with MLAnomalyDetector

The typical integration is as a `gray_zone_verifier` in `MLAnomalyDetector`:

```python
import anthropic
from axor_classifier_simple import MLAnomalyDetector
from axor_classifier_llm import LLMAnomalyVerifier

verifier = LLMAnomalyVerifier(client=anthropic.AsyncAnthropic())

detector = MLAnomalyDetector(
    gray_zone_verifier=verifier,
    gray_zone_threshold=0.50,   # escalate when score >= 0.50 in suspicious range
)

result = await detector.score(window=intents)
```

Escalation logic in `MLAnomalyDetector`:

| Score | Class | LLM called? |
|-------|-------|-------------|
| `< 0.40` | `NORMAL` | No |
| `[0.40, 0.50)` | `SUSPICIOUS` | No (below threshold) |
| `[0.50, 0.75)` | `SUSPICIOUS` | Yes — verifier result returned |
| `>= 0.75` | `CRITICAL` | No |

If the LLM call raises any exception, `MLAnomalyDetector` falls back to the ML-derived score and logs a warning.

---

## What the LLM sees

Each `NormalizedIntent` in the window is formatted as a single line:

```
tool=bash op=execute_generated_code target=workdir dest=none prov=repo flow=local_to_local [executes_generated_code, after_external_read]
```

The flags in brackets (`reads_secret`, `writes_outside_workdir`, etc.) appear only when set.

The system prompt instructs Claude to evaluate **behavioral patterns only**:

- External read → secret access → outbound network → `critical`
- Cloud metadata probe → `critical`
- Docker socket access → `critical`
- Unexpected tool class for stated task → `suspicious`
- Normal coding / research patterns → `normal`

Claude responds with JSON only:

```json
{
  "score": 0.82,
  "class": "critical",
  "reasons": ["secret_access_after_external_read", "network_after_secret"]
}
```

Malformed responses are caught: `_parse_response` returns `score=0.5 / SUSPICIOUS / ("verifier_parse_error",)` and logs the exception at `WARNING` level.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `client` | required | `anthropic.AsyncAnthropic` instance |
| `model` | `claude-haiku-4-5-20251001` | Anthropic model ID |
| `max_tokens` | `256` | Max tokens for the verifier response |

---

## AnomalyResult contract

Defined in `axor_core.contracts.anomaly`:

```python
@dataclass
class AnomalyResult:
    score: float              # 0.0 – 1.0 risk score
    cls: AnomalyClass         # NORMAL | SUSPICIOUS | CRITICAL
    reasons: tuple[str, ...]  # human-readable trigger reasons
```

Score thresholds:

| Class | Range |
|-------|-------|
| `NORMAL` | `[0.0, 0.40)` |
| `SUSPICIOUS` | `[0.40, 0.75)` |
| `CRITICAL` | `[0.75, 1.0]` |

---

## Development

```bash
git clone https://github.com/Bucha11/axor-classifier-llm
cd axor-classifier-llm
pip install -e ".[dev]"
pytest tests/
```

Tests mock the Anthropic client and do not make real API calls.

---

## License

MIT
