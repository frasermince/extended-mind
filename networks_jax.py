import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
import numpy as np


# def layer_init(key, shape, std=np.sqrt(2), bias_const=0.0):
#     # JAX version of layer initialization
#     w_key, b_key = jax.random.split(key)
#     w = std * jax.random.orthogonal(w_key, shape=shape)
#     b = jnp.full(shape[-1], bias_const)
#     return w, b


class Network(nn.Module):
    """Convolutional network for Q-values"""

    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = jnp.expand_dims(x, axis=-1)
        # x = jnp.transpose(x, (0, 3, 1, 2))
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x
