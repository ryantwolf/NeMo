import re

import pytest
from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import DEFAULT_CONVERSION_DICT

from nemo.export.tensorrt_llm import TensorRTLLM


def test_get_nemo_to_trtllm_conversion_dict_on_nemo_model():
    dummy_state = object()
    model_state_dict = {
        'model.embedding.word_embeddings.weight': dummy_state,
        'model.decoder.layers.0.self_attention.linear_proj.weight': dummy_state,
    }
    nemo_model_conversion_dict = TensorRTLLM.get_nemo_to_trtllm_conversion_dict(model_state_dict)

    # Check that every key starts with 'model.' and not 'model..' by using a regex
    # This pattern ensures:
    #   - The key starts with 'model.'
    #   - Immediately after 'model.', there must be at least one character that is NOT a '.'
    #     (preventing the 'model..' scenario)
    pattern = re.compile(r'^model\.[^.].*')
    for key in nemo_model_conversion_dict.keys():
        assert pattern.match(key), f"Key '{key}' does not properly start with 'model.'"


def test_get_nemo_to_trtllm_conversion_dict_on_mcore_model():
    dummy_state = object()
    model_state_dict = {
        'embedding.word_embeddings.weight': dummy_state,
        'decoder.layers.0.self_attention.linear_proj.weight': dummy_state,
    }
    nemo_model_conversion_dict = TensorRTLLM.get_nemo_to_trtllm_conversion_dict(model_state_dict)

    # This is essentially a no-op
    assert nemo_model_conversion_dict == DEFAULT_CONVERSION_DICT
