from traceguard.json_utils import extract_json_object


def test_extract_json_object_from_fenced_response():
    raw = '```json\n{"label":"safe","confidence":0.7}\n```'
    assert extract_json_object(raw)["label"] == "safe"


def test_extract_json_object_from_prefixed_response():
    raw = 'Here is the report: {"label":"unsafe","evidence_steps":[3,4]} done.'
    assert extract_json_object(raw)["evidence_steps"] == [3, 4]

