# to run these, run 
# pytest test/test-validator.py

from guardrails import Guard
from validator import CompetitorCheck

# We use 'refrain' as the validator's fail action,
#  so we expect failures to always result in a guarded output of None
# Learn more about corrective actions here:
#  https://www.guardrailsai.com/docs/concepts/output/#%EF%B8%8F-specifying-corrective-actions
competitors_list = [
    "Acorns",
    "Citigroup",
    "Citi",
    "Fidelity Investments",
    "Fidelity",
    "JP Morgan Chase and company",
    "JP Morgan",
    "JP Morgan Chase",
    "JPMorgan Chase",
    "Chase" "M1 Finance",
    "Stash Financial Incorporated",
    "Stash",
    "Tastytrade Incorporated",
    "Tastytrade",
    "ZacksTrade",
    "Zacks Trade",
]

guard = Guard.from_string(validators=[CompetitorCheck(competitors=competitors_list, on_fail="refrain")])


def test_pass():
  test_output = "HomeDepot is not a competitor"
  raw_output, guarded_output, *rest = guard.parse(test_output)
  assert(guarded_output is test_output)

def test_fail():
  test_output = "Acorns, with its extensive global network, has become a powerhouse in the banking sector, catering to the needs of millions across different countries."
  raw_output, guarded_output, *rest = guard.parse(test_output)
  assert(guarded_output is None)
