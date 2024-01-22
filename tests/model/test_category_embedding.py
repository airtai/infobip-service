from infobip_service.model import build_embedding_layer_category

def test_build_embedding_layer_category():
    layer = build_embedding_layer_category(
        "DefinitionId", ["Entered_flow", "Exited_flow"], embedding_dim=10
    )
    actual = layer(["Entered_flow", "Exited_flow"])
    assert actual.shape == (2, 10)
