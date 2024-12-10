# to run these, run 
# pytest test/test-validator.py

from guardrails import Guard
from guardrails.validators import PassResult, FailResult
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


def test_pass():
    v = CompetitorCheck(
        competitors=[
            "JP Morgan Chase",
            "Alphabet Incorporated",
            "Google Inc",
            "Apple Inc",
        ],
        use_local=True,
        threshold=0.5,
    )
    assert isinstance(v.validate("This must be fine."), PassResult)
    assert isinstance(v.validate("I use Bing."), PassResult)
    assert isinstance(v.validate("My cat chased my dog."), PassResult)
    assert isinstance(v.validate("I baked an apple pie."), PassResult)
    assert isinstance(v.validate("I searched the internet for info."), PassResult)


def test_fail():
    v = CompetitorCheck(
        competitors=[
            "JP Morgan Chase",
            "Alphabet Incorporated",
            "Google Inc",
            "Apple Inc",
        ],
        use_local=True,
        threshold=0.5,
    )
    assert isinstance(v.validate("I bought an iPhone."), FailResult)
    assert isinstance(v.validate("I use Google Photos."), FailResult)
    assert isinstance(v.validate("My mortgage comes from Chase Bank."), FailResult)
    assert isinstance(v.validate("I googled how to bake a pie on my Macbook."), FailResult)
