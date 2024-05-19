# Overview

| Developed by | Guardrails AI |
| Date of development | Feb 15, 2024 |
| Validator type | Brand risk, QA, chatbots |
| Blog |  |
| License | Apache 2 |
| Input/Output | Output |

## Description

### Intended Use
This validator ensures that no competitors for an organization are being named. In order to use this validator, you need to provide a list of competitors that you don’t want to name.

### Requirements

* Dependencies:
    - guardrails-ai>=0.4.0
    - nltk

## Installation

```bash
$ guardrails hub install hub://guardrails/competitor_check
```

## Usage Examples

### Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails import Guard
from guardrails.hub import CompetitorCheck


# Setup Guard
guard = Guard().use(CompetitorCheck, ["Apple", "Samsung"], "exception")

response = guard.validate(
    "The apple doesn't fall far from the tree."
)  # Validator passes

try:
    response = guard.validate("Apple just released a new iPhone.")  # Validator fails
except Exception as e:
    print(e)
```
Output:
```console
Validation failed for field with errors: Found the following competitors: [['Apple']]. Please avoid naming those competitors next time
```

### Validating JSON output via Python

In this example, we apply the validator to a string that is a field within a Pydantic object.

```python
# Import Guard and Validator
from pydantic import BaseModel, Field
from guardrails.hub import CompetitorCheck
from guardrails import Guard

# Initialize Validator
val = CompetitorCheck(competitors=["Apple", "Samsung"], on_fail="exception")


# Create Pydantic BaseModel
class MarketingCopy(BaseModel):
    product_name: str
    product_description: str = Field(
        description="Description about the product", validators=[val]
    )


# Create a Guard to check for valid Pydantic output
guard = Guard.from_pydantic(output_class=MarketingCopy)

# Run LLM output generating JSON through guard
try:
    guard.parse(
        """
        {
            "product_name": "Galaxy S23+",
            "product_description": "Samsung's latest flagship phone with 5G capabilities"
        }
        """
    )
except Exception as e:
    print(e)
```
Output:
```console
Validation failed for field with errors: Found the following competitors: [['Samsung']]. Please avoid naming those competitors next time
```

# API Reference

**`__init__(self, competitors, on_fail="noop")`**
<ul>
Initializes a new instance of the Validator class.

**Parameters**
- **`competitors`** *(List[str])*: List of names of competitors to avoid.
- **`on_fail`** *(str, Callable)*: The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.
</ul>
<br/>

**`validate(self, value, metadata={}) → ValidationResult`**
<ul>
Validates the given `value` using the rules defined in this validator, relying on the `metadata` provided to customize the validation process. This method is automatically invoked by `guard.parse(...)`, ensuring the validation logic is applied to the input data.

Note:

1. This method should not be called directly by the user. Instead, invoke `guard.parse(...)` where this method will be called internally for each associated Validator.
2. When invoking `guard.parse(...)`, ensure to pass the appropriate `metadata` dictionary that includes keys and values required by this validator. If `guard` is associated with multiple validators, combine all necessary metadata into a single dictionary.

**Parameters**
- **`value`** *(Any)*: The input value to validate.
- **`metadata`** *(dict)*: A dictionary containing metadata required for validation. No additional metadata keys are needed for this validator.

</ul>
