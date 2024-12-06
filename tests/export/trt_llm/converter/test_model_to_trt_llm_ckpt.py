import pytest
from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import get_layer_prefix


@pytest.mark.parametrize(
    'input_layer_names,expected_model_prefix',
    [
        (
            [
                'model.embedding.word_embeddings.weight',
                'model.decoder.layers.0.self_attention.linear_proj.weight',
                'model.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight',
                'model.decoder.layers.0.self_attention.linear_qkv.weight',
                'model.decoder.layers.0.mlp.linear_fc1.layer_norm_weight',
                'model.decoder.layers.0.mlp.linear_fc1.weight',
                'model.decoder.layers.0.mlp.linear_fc2.weight',
            ],
            'model.',
        )
    ],
)
def test_get_layer_prefix_is_mcore(input_layer_names, expected_model_prefix):
    model_prefix, _ = get_layer_prefix(input_layer_names, is_mcore=True)
    assert model_prefix == expected_model_prefix
