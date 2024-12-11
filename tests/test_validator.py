# to run these, run 
# pytest test/test-validator.py

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
            "Chase",
            "Google",
            "Apple",
        ],
        use_local=True,
        threshold=0.5,
        on_fail='noop',
    )
    assert isinstance(v.validate("This must be fine."), PassResult)
    assert isinstance(v.validate("I use Bing."), PassResult)
    assert isinstance(v.validate("My cat chased my dog, which is funny because his name is Chase."), PassResult)
    assert isinstance(v.validate("I baked an apple pie."), PassResult)
    assert isinstance(v.validate("I searched the internet for info."), PassResult)


def test_fail():
    v = CompetitorCheck(
        competitors=[
            "Chase",
            "Google",
            "Apple",
        ],
        use_local=True,
        threshold=0.5,
        on_fail='noop',
    )
    assert isinstance(v.validate("I bought an Apple iPhone."), FailResult)
    assert isinstance(v.validate("Android is Google's phone offering."), FailResult)
    assert isinstance(v.validate("My mortgage comes from Chase."), FailResult)
    res = v.validate("I googled how to bake a pie on my Apple MacBook.")
    print(res.outcome)
    assert isinstance(res, FailResult)
