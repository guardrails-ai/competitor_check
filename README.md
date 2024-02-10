## Details

| Developed by | Guardrails AI |
| --- | --- |
| Date of development | Feb 15, 2024 |
| Validator type | Brand risk, QA, chatbots |
| Blog |  |
| License | Apache 2 |
| Input/Output | Output |

## Description

This validator ensures that no competitors for an organization are being named. In order to use this validator, you need to provide a list of competitors that you don’t want to name.

## Example Usage Guide

### Installation

```bash
$ guardrails hub install hub://guardrails/competitor-check
```

### Initialization

```python
from guardrails.hub import CompetitorCheck
competitor_check = CompetitorCheck(
	competitors_list=["name1", "name2",]
	on_fail="noop"
)

# Create Guard with Validator
guard = Guard.from_string(
    validators=[competitor_check, ...],
)
```

### Invocation

```python
guard("Some LLM output")
```

## Intended use

- Primary intended uses: Primarily for QA and chatbots in an enterprise setting
- Out-of-scope use cases: na

## Expected deployment metrics

|  | CPU | GPU |
| --- | --- | --- |
| Latency |  | - |
| Memory |  | - |
| Cost |  | - |
| Expected quality |  | - |

## Resources required

- Dependencies: `nltk`
- Foundation model access keys: na
- Compute: na

## Validator Performance

### Evaluation Dataset

### Model Performance Measures

| Accuracy |  |
| --- | --- |
| F1 Score |  |

### Decision thresholds
