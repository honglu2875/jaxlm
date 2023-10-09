import jax
import torch
from _hf_mistral import MistralModel  # for debugging
from transformers import AutoTokenizer, MistralConfig, MistralModel

from mistral_jax import MistralModel as MistralModelJax
from mistral_jax.utils import torch_to_jax_states


def _forward_pass(model, model_jax, inputs, inputs_jax):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    key = jax.random.PRNGKey(0)
    params = model_jax.init(key, inputs_jax["input_ids"])
    params = {**params, **torch_to_jax_states(model)}
    outputs_jax = model_jax.apply(
        params,
        inputs_jax["input_ids"],
        attention_mask=inputs_jax["attention_mask"],
        mutable=["cache"],
        output_hidden_states=True,
    )
    return outputs, outputs_jax


def test_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    config = MistralConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
    )
    model = MistralModel(config)
    model_jax = MistralModelJax(config)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    inputs_jax = tokenizer("Hello, my dog is cute", return_tensors="jax")

    outputs, outputs_jax = _forward_pass(model, model_jax, inputs, inputs_jax)

    for i in range(len(outputs.hidden_states)):
        hidden = outputs.hidden_states[i].numpy()
        hidden_jax = outputs_jax[0].hidden_states[i]
        assert jax.numpy.allclose(hidden, hidden_jax, atol=1e-3)

    # With attention mask
    inputs = {
        **inputs,
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.int32),
    }
    inputs_jax = {
        **inputs_jax,
        "attention_mask": jax.numpy.array(
            [[1, 1, 1, 0, 0, 0, 0]], dtype=jax.numpy.int32
        ),
    }

    outputs, outputs_jax = _forward_pass(model, model_jax, inputs, inputs_jax)

    for i in range(len(outputs.hidden_states)):
        hidden = outputs.hidden_states[i].numpy()
        hidden_jax = outputs_jax[0].hidden_states[i]
        assert jax.numpy.allclose(hidden, hidden_jax, atol=1e-3)
