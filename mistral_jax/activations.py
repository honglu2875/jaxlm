import jax


# Directly adapted from HuggingFace's transformers
ACT2FN = {
    "gelu": jax.nn.gelu,
    #"gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    #"gelu_fast": FastGELUActivation,
    #"gelu_new": NewGELUActivation,
    #"gelu_python": (GELUActivation, {"use_gelu_python": True}),
    #"gelu_pytorch_tanh": PytorchGELUTanh,
    #"gelu_accurate": AccurateGELUActivation,
    #"laplace": LaplaceActivation,
    "linear": jax.nn.leaky_relu,
    #"mish": MishActivation,
    #"quick_gelu": QuickGELUActivation,
    "relu": jax.nn.relu,
    #"relu2": ReLUSquaredActivation,
    #"relu6": nn.ReLU6,
    "sigmoid": jax.nn.sigmoid,
    "silu": jax.nn.silu,
    "swish": jax.nn.swish,
    "tanh": jax.nn.tanh,
}
