from validator import CompetitorCheck


def test_prefilter_exact_match():
    # Make sure that the prefilter will correctly flag when a sentence has a competitor.
    competitors = ["abc", "ghi"]
    sentences = [
        "abc",
        "abcdef",
        "ab def",
        "a def b asdf c",
        "ghist of it",
    ]
    potential_matches = CompetitorCheck.exact_match(sentences, competitors)
    assert potential_matches[0]
    assert potential_matches[1]
    assert not potential_matches[2]
    assert not potential_matches[3]
    assert potential_matches[4]
