import pytest

from recruit_intel.models import defence


@pytest.fixture(autouse=True)
def no_trained_defence_model(monkeypatch, request):
    """Unit tests run on hand-built databases: a defensive model trained on
    the real one (data/models/) must not fill their tables."""
    if "uses_trained_models" not in request.keywords:
        monkeypatch.setattr(defence, "load", lambda path=defence.PATH: None)
