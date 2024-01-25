import pytest


@pytest.fixture()
def patch_envs(monkeypatch):
    monkeypatch.setenv("KAFKA_HOSTNAME", "localhost")
    monkeypatch.setenv("KAFKA_PORT", "9092")

    temp_username = "temp_username"
    monkeypatch.setenv("USERNAME", temp_username)
