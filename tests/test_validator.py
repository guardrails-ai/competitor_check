import pytest
from guardrails import Guard
from guardrails.hub import CompetitorCheck

# Setup Guard with CompetitorCheck validator
guard = Guard().use(
    CompetitorCheck, ["Apple", "Samsung"], "exception"
)

# Test passing response (no competitor mentioned)
def test_competitor_check_pass():
    response = guard.validate("The apple doesn't fall far from the tree.")
    assert response.validation_passed is True

# Test failing response (competitor mentioned)
def test_competitor_check_fail():
    with pytest.raises(Exception) as e:
        guard.validate("Apple just released a new iPhone.")
    assert "Validation failed for field with errors:" in str(e.value)
