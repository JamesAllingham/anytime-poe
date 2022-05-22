from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array
import distrax


class Cls_Ens(nn.Module):
    """A standard classification ensemble."""
    size: int
    make_net: Callable[[], nn.Module]
    weights_init: Callable = initializers.ones
    logscale_init: Callable = initializers.zeros
    learn_weights: bool = False

    def setup(self):
        self.nets = [self.make_net() for _ in range(self.size)]
        self.weights = self.param(
            'weights',
            self.weights_init,
            (self.size,)
        )

    def __call__(
        self,
        x: Array,
        train: bool = False,
    ) -> Array:
        logits = jnp.stack([net(x, train=train) for net in self.nets], axis=0)
        probs = nn.softmax(self.weights, axis=0)[:, jnp.newaxis]  # (M, 1)

        return logits, probs


def make_Cls_Ens_loss(
    model: Cls_Ens,
    x_batch: Array,
    y_batch: Array,
    train: bool = True,
) -> Callable:
    """Creates a loss function for training a std Ens."""
    def batch_loss(params, state):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            (ens_logits, probs), new_state = model.apply(
                {"params": params, **state}, x, train=train,
                mutable=list(state.keys()) if train else {},
            )

            def nll(y, logits):
                return  -1 * distrax.Categorical(logits).log_prob(y)

            nlls = jax.vmap(nll, in_axes=(None, 0))(y, ens_logits)
            loss = (nlls * probs).sum(axis=0)

            error = jnp.argmax((ens_logits * probs).sum(axis=0)) != y

            return loss, error, new_state

        # broadcast over batch and take mean
        loss_for_batch, errors_for_batch, new_state = jax.vmap(
            loss_fn, out_axes=(0, 0, None), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return loss_for_batch.mean(axis=0), (errors_for_batch.mean(axis=0), new_state)

    return batch_loss
