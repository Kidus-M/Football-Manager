from recruit_intel.identity.names import normalize, tokens


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Altay Bayındır") == "altay bayindir"
    assert normalize("Martin Ødegaard") == "martin odegaard"
    assert normalize("Jean-Philippe Mateta") == "jean philippe mateta"


def test_tokens():
    assert tokens("Bruno Borges Fernandes") >= {"bruno", "fernandes"}
