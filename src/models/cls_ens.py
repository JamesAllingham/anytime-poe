from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array
import distrax

from src.models.common import get_agg_fn
from src.models.resnet import ResNet


KwArgs = Mapping[str, Any]


class Cls_Ens(nn.Module):
    """A standard classification ensemble."""
    size: int
    net: KwArgs
    weights_init: Callable = initializers.ones
    logscale_init: Callable = initializers.zeros
    learn_weights: bool = False

    def setup(self):
        self.nets = [ResNet(**self.net) for _ in range(self.size)]
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
        probs = nn.softmax(self.weights, axis=0)  # (M,)

        def nll(y, logits):
            return  -1 * distrax.Categorical(logits).log_prob(y)

        nlls = jax.vmap(nll, in_axes=(None, 0))(y, ens_logits)
        loss = (nlls * probs).sum(axis=0)

        pred = nn.softmax(ens_logits.mean(axis=0))
        err = y != jnp.argmax(pred, axis=0)

        return loss, err

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
    aggregation: str = 'mean',
) -> Callable:
    """Creates a loss function for training a std Ens."""
    def batch_loss(params, state, rng):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            (loss, err), new_state = model.apply(
                {"params": params, **state}, x, y, train=train,
                mutable=list(state.keys()) if train else {},
                rngs={'dropout': rng},
            )

            return loss, new_state, err

        # broadcast over batch and aggregate
        agg = get_agg_fn(aggregation)
        loss_for_batch, new_state, err_for_batch = jax.vmap(
            loss_fn, out_axes=(0, None, 0), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return agg(loss_for_batch, axis=0), (new_state, agg(err_for_batch, axis=0))

    return batch_loss
