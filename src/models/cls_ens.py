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
        weights = self.param(
            'weights',
            self.weights_init,
            (self.size,)
        )
        self.weights = weights if self.learn_weights else jax.lax.stop_gradient(weights)

    def __call__(
        self,
        x: Array,
        y: int,
        train: bool = False,
    ) -> Array:
        ens_logits = jnp.stack([net(x, train=train) for net in self.nets], axis=0)  # (M, O)
        probs = nn.softmax(self.weights, axis=0)[:, jnp.newaxis]  # (M, 1)

        def nll(y, logits):
            return  -1 * distrax.Categorical(logits).log_prob(y)

        nlls = jax.vmap(nll, in_axes=(None, 0))(y, ens_logits)
        loss = (nlls * probs).sum(axis=0)

        return loss

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
    ) -> Array:
        ens_logits = jnp.stack([net(x, train=train) for net in self.nets], axis=0)  # (M, O)
        probs = nn.softmax(self.weights, axis=0)[:, jnp.newaxis]  # (M, 1)

        logits = (probs * ens_logits).sum(axis=0)  # (O,)
        preds = nn.softmax(logits)

        if return_ens_preds:
            return preds, nn.softmax(ens_logits, axis=-1)
        else:
            return preds


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
            loss, new_state = model.apply(
                {"params": params, **state}, x, train=train,
                mutable=list(state.keys()) if train else {},
            )

            return loss, new_state

        # broadcast over batch and take mean
        loss_for_batch, new_state = jax.vmap(
            loss_fn, out_axes=(0, None), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return loss_for_batch.mean(axis=0), new_state

    return batch_loss