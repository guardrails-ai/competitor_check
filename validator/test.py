# Import Guard and Validator
from guardrails import Guard
from CompetitorCheck import CompetitorCheck


# Setup Guard
guard = Guard().use(
    CompetitorCheck, ["Apple", "Samsung"], "exception", use_local=True,
)
response = guard.validate("apple just released a new iPhone.")  # Validator fails

print(guard.error_spans_in_output())