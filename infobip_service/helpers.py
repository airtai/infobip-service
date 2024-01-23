from os import environ
from typing import Any, Dict

from aiokafka.helpers import create_ssl_context


def get_aio_kafka_config() -> Dict[str, Any]:
    kafka_server_url = environ["KAFKA_HOSTNAME"]
    kafka_server_port = environ["KAFKA_PORT"]

    kafka_bootstrap_servers = (
        f":{kafka_server_port},".join(kafka_server_url.split(","))
        + f":{kafka_server_port}"
    )
    return {
        "bootstrap_servers": kafka_bootstrap_servers,
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": environ["KAFKA_SASL_MECHANISM"],
        "sasl_plain_username": environ["KAFKA_API_KEY"],
        "sasl_plain_password": environ["KAFKA_API_SECRET"],
        "ssl_context": create_ssl_context(),
    }
