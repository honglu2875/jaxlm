# mistral_jax
(unofficial) Mistral model in JAX

## Quickstart

```bash
pip install -e .
```
In Python:
```python
import jax
# Still works with HuggingFace's tokenizer and config
from transformers import AutoTokenizer, MistralForCausalLM

from mistral_jax import MistralForCausalLM as MistralForCausalLMJAX
from mistral_jax.utils import torch_to_jax_states
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Tokenize the prompt
inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

# Initialize the JAX model
model_jax = MistralForCausalLMJAX(model.config)

# Get the initial parameters (esp. for the mutable variables)
key = jax.random.PRNGKey(0)
# If len(jax.devices()) > 1, the input is automatically sharded (2, ...) as ("data", "model")
inputs = model_jax.prepare_input(inputs["input_ids"])

# Converted state dict from the PyTorch model, and possibly shard the params (if devices > 1)
params = model_jax.get_params(weights=torch_to_jax_states(model.state_dict()))
```
To obtain individual logit outputs and kv-cache:
```python
outputs, mutable_vars = model_jax.apply(
    params,
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    mutable=("cache",),
    output_hidden_states=True,
)
```
To perform a completion:
```python
out_jax = model_jax.generate(
    params, 
    inputs_jax["input_ids"], 
    do_sample=True, 
    max_length=100
)
completion = tokenizer.batch_decode(out_jax)
```