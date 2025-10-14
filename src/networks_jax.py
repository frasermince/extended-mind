from flax import linen as nn
import dataclasses


# def layer_init(key, shape, std=np.sqrt(2), bias_const=0.0):
#     # JAX version of layer initialization
#     w_key, b_key = jax.random.split(key)
#     w = std * jax.random.orthogonal(w_key, shape=shape)
#     b = jnp.full(shape[-1], bias_const)
#     return w, b


class Network(nn.Module):
    """Convolutional network for Q-values"""

    action_dim: int
    feature_dims: list[int] = dataclasses.field(default_factory=lambda: [120, 84])

    @nn.compact
    def __call__(self, x):
        x = x / 255.0

        x = x.reshape((x.shape[0], -1))
        for feature_dim in self.feature_dims:
            x = nn.Dense(feature_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x
