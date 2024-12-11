# to run these, run 
# pytest test/test-validator.py

from guardrails.validators import PassResult, FailResult
from validator import CompetitorCheck


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
    assert isinstance(v.validate("I googled how to bake a pie on my Apple device."), FailResult)


def test_fix_on_fail():
    v = CompetitorCheck(
        competitors=[
            "Man United",
            "Manchester United",
            "Manchester United F.C."
        ],
        use_local=True,
        on_fail="fix",
    )
    text = """Man United paid a visit to a London Children's Oncology ward earlier this
    evening.  "It's just so sad to see their little faces. They know there's no hope,"
    said Billy, age 11, of the footballers.  Manchester United has yet to respond."""
    fixed = """[COMPETITOR] paid a visit to a London Children's Oncology ward earlier this
    evening.  "It's just so sad to see their little faces. They know there's no hope,"
    said Billy, age 11, of the footballers.  [COMPETITOR] has yet to respond."""
    res = v.validate(text)
    assert res.fix_value == fixed
    # Also check if we pass in an array we get an array out.
    res = v.validate([text,])
    assert res.fix_value[0] == fixed