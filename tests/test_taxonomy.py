from traceguard.taxonomy import FAILURE_MODES, HARM_TYPES, RISK_SOURCES


def test_agentdog_taxonomy_counts():
    assert len(RISK_SOURCES) == 8
    assert len(FAILURE_MODES) == 14
    assert len(HARM_TYPES) == 10

