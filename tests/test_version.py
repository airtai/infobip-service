import infobip_service


def test_version() -> None:
    assert infobip_service.__version__ is not None
