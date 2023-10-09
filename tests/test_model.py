import jax
import torch
from transformers import MistralModel, MistralConfig, AutoTokenizer
from mistral_jax._torch import MistralModel
from mistral_jax import MistralModel as MistralModelJax
from mistral_jax.utils import torch_to_jax_states


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

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    key = jax.random.PRNGKey(0)
    params = model_jax.init(key, inputs_jax["input_ids"])
    params = {**params, **torch_to_jax_states(model)}
    outputs_jax = model_jax.apply(params, inputs_jax["input_ids"], mutable=["cache"], output_hidden_states=True)


    for i in range(len(outputs.hidden_states)):
        hidden = outputs.hidden_states[i].numpy()
        hidden_jax = outputs_jax[0].hidden_states[i]
        print(hidden - hidden_jax)
        assert jax.numpy.allclose(hidden, hidden_jax, atol=1e-3)
