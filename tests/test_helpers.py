import ssl

from infobip_service.helpers import get_aio_kafka_config


def test_get_aio_kafka_config(monkeypatch):
    monkeypatch.setenv("KAFKA_HOSTNAME", "kafka")
    monkeypatch.setenv("KAFKA_PORT", "9092")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
    monkeypatch.setenv("KAFKA_API_KEY", "test")
    monkeypatch.setenv("KAFKA_API_SECRET", "test")

    actual = get_aio_kafka_config()
    ssl_context = actual.pop("ssl_context")
    assert isinstance(ssl_context, ssl.SSLContext)

    expected = {
        "bootstrap_servers": "kafka:9092",
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "SCRAM-SHA-256",
        "sasl_plain_username": "test",
        "sasl_plain_password": "test",
    }
    assert actual == expected, actual
