# Import Guard and Validator
from guardrails import Guard
from guardrails.hub import CompetitorCheck


# Setup Guard
guard = Guard().use(
    CompetitorCheck(["Apple", "Samsung"], use_local=False, on_fail="exception")
)

response = guard.validate(
    "The apple doesn't fall far from the tree."
)  # Validator passes

try:
    response = guard.validate("Apple just released a new iPhone.")  # Validator fails
except Exception as e:
    print(e)