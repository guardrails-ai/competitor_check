import guardrails as gd
from guardrails.hub import CompetitorCheck

# Create a Guard class
guard = gd.Guard().use(
    CompetitorCheck(competitors=["Apple", "Samsung"], use_local=False, on_fail="exception", validation_endpoint="http://127.0.0.1:8000/validate"),
)


try:
#     response = guard.validate(
#     "The apple doesn't fall far from the tree."
# )  # Validator passes
    response = guard.validate("apple and samsung are great man.")  # Validator fails
    print('passed')
except Exception as e:
    print(e)