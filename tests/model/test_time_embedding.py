import numpy as np
import torch

from infobip_service.model import build_embedding_layer_time
from infobip_service.model.time_embedding import (
    PERIODS,
    PERIODS_SECONDS,
    create_period_normalization,
    expand_time_feature,
    expand_time_features,
)


def test_expand_time_feature():
    x = torch.Tensor([1662994700.0, 1682329500.0, 1670535600.0])

    for period in PERIODS:
        actual = expand_time_feature(x, period)
        assert (actual >= 0).numpy().all()
        assert (actual < PERIODS_SECONDS[period]).numpy().all()


def test_expand_time_features():
    d = {
        "SIGNUP_DATE": torch.Tensor([1662234500.0, 1682543600.0, 1670437700.0]),
        "COMPLETE_REGST_EVENT_DATE": torch.Tensor(
            [1662994700.0, 1682329500.0, 1670535600.0]
        ),
    }

    actual = expand_time_features(d)
    assert len(actual.keys()) == 2 * (len(PERIODS))
    actual


def test_create_period_normalization():
    normalization = create_period_normalization(name="SIGNUP_DATE", period="day")

    xs = torch.Tensor(np.arange(0, PERIODS_SECONDS["day"], 1).reshape((-1, 1)))
    actual = normalization(xs)
    np.testing.assert_almost_equal(actual.numpy().min(), -1, decimal=6)
    np.testing.assert_almost_equal(actual.numpy().max(), 1, decimal=4)


def test_build_embedding_layer_time():
    layer = build_embedding_layer_time(
        "COMPLETE_REGST_EVENT_DATE", mean=34.0, std=8.0, output_dim=10
    )
    actual = layer(torch.Tensor([25.0, 42.0]))
    assert actual.shape == (2, 10)


def test_build_embedding_layer_time_batch():
    layer = build_embedding_layer_time(
        "COMPLETE_REGST_EVENT_DATE", mean=34.0, std=8.0, output_dim=10
    )
    actual = layer(torch.Tensor([[25.0, 42.0], [25.0, 42.0], [25.0, 42.0]]))
    assert actual.shape == (3, 2, 10)
