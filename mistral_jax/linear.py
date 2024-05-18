#  Copyright 2024 Honglu Fan
#  This file is based on code by the authors denoted below and has been modified from its original version.
#
#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# This file has been modified from its original version
# Link: https://github.com/google/maxtext/blob/4f3a0d3cf8509d05ce040e35d88ea7bf57797945/MaxText/layers/linears.py
#

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from .types import PRNGKey, Array, Config, DType, Shape
from typing import Any, Callable, Iterable, Sequence, Tuple, Union, Optional



def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneral(nn.Module):
    """A linear transformation with flexible axes.

    Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    use_bias: whether to add bias in linear transformation
    """

    features: Union[Iterable[int], int]
    axis: Union[Iterable[int], int] = -1
    weight_dtype: DType = jnp.float32
    dtype: DType = jnp.float32
    kernel_init: Callable = nn.initializers.variance_scaling
    kernel_init_args: tuple = (1.0, "fan_in", "truncated_normal")
    with_logical_partitioning: bool = True
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = False

    def setup(self):
        # wrap over init function in order to receive in_axis and out_axis
        def init_fn(key: PRNGKey, shape: Shape, dtype: DType, in_axis: int, out_axis: int):
            fn = self.kernel_init(*self.kernel_init_args, in_axis=in_axis, out_axis=out_axis)
            if self.with_logical_partitioning:
                fn = nn.with_logical_partitioning(fn, self.kernel_axes)
            return fn(key, shape, dtype)
        self.wrapped_kernel_init = init_fn

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """

        def compute_dot_general(inputs, kernel, axis, contract_ind):
            """Computes a dot_general operation."""
            dot_general = lax.dot_general
            return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)

        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
        kernel_in_axis = np.arange(len(axis))
        kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
        kernel = self.param(
            "kernel",
            self.wrapped_kernel_init,
            kernel_shape,
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
        )
        kernel = jnp.asarray(kernel, self.dtype)

        contract_ind = tuple(range(0, len(axis)))
        output = compute_dot_general(inputs, kernel, axis, contract_ind)

        if self.use_bias:
            bias_axes, bias_shape = self.kernel_axes[-len(features) :], kernel_shape[-len(features) :]
            bias = self.param(
                "bias",
                nn.with_logical_partitioning(bias_init, bias_axes),
                bias_shape,
                self.weight_dtype,
            )
            bias = jnp.asarray(bias, self.dtype)
            output += bias
        return output
