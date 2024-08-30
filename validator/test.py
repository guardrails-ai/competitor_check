# Import Guard and Validator
from guardrails import Guard
from guardrails.hub import CompetitorCheck


# Setup Guard
guard = Guard().use(
    CompetitorCheck, ["Apple", "Samsung"], "noop", use_local=False,
)

response = guard.validate(
    "samsung is a tech company that makes phones, computers, and tablets."
)  

# try:
#     response = guard.validate("Apple just released a new iPhone.")  # Validator fails
# except Exception as e:
#     print(e)

print(response)