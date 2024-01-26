import torch

from infobip_service.model import build_embedding_layer_category


def test_build_embedding_layer_category():
    layer = build_embedding_layer_category("DefinitionId", 3, embedding_dim=10)
    actual = layer(torch.Tensor([0, 2]).to(torch.int64))
    assert actual.shape == (2, 10)


def test_build_embedding_layer_category_batch():
    layer = build_embedding_layer_category("DefinitionId", 3, embedding_dim=10)
    actual = layer(torch.Tensor([[0, 2], [0, 2], [0, 2]]).to(torch.int64))
    assert actual.shape == (3, 2, 10)
