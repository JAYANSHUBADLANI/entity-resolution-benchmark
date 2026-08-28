from src.er.text import normalise, tokens


def test_normalise_strips_case_punctuation_and_accents():
    assert normalise("ZoneAlarm(R) Internet-Security!") == "zonealarm r internet security"
    assert normalise("Café  Münster") == "cafe munster"


def test_normalise_treats_missing_markers_as_empty():
    assert normalise(None) == ""
    assert normalise(float("nan")) == ""
    assert normalise("NaN") == ""


def test_tokens_are_deduplicatable_words():
    assert tokens("The Print Shop 22") == ["the", "print", "shop", "22"]
    assert tokens("") == []
