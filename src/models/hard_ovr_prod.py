from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array

from src.models.common import get_agg_fn
from src.models.resnet import ResNet


KwArgs = Mapping[str, Any]


def hardened_ovr_ll(y_1hot, logits, T):
    σ = nn.sigmoid(T * logits)
    return jnp.sum(y_1hot * jnp.log(σ) + (1 - y_1hot) * jnp.log(1 - σ), axis=0)


class Hard_OvR_Ens(nn.Module):
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
        β: int = 1,
    ) -> Array:
        ens_logits = jnp.stack([net(x, train=train) for net in self.nets], axis=0)  # (Μ, Ο)
        probs = nn.softmax(self.weights, axis=0)  # (M,)

        n_classes = self.net['out_size']

        def product_logprob(y):
            y_1hot = jax.nn.one_hot(y, n_classes)
            return jnp.sum(probs * jax.vmap(hardened_ovr_ll, in_axes=(None, 0, None))(y_1hot, ens_logits, β))

        ys = jnp.arange(n_classes)
        Z = jnp.sum(jnp.exp(jax.vmap(product_logprob)(ys)), axis=0)

        nll = -(product_logprob(y) - jnp.log(Z + 1e-36))

        return nll#, y, product_logprob(y), jnp.log(Z + 1e-36)

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
    ) -> Array:
        ens_logits = jnp.stack([net(x, train=train) for net in self.nets], axis=0)  # (M, O)
        ens_preds = jnp.round(nn.sigmoid(ens_logits))

        preds = ens_preds.prod(axis=0)

        if return_ens_preds:
            return preds, ens_preds
        else:
            return preds


def make_Hard_OvR_Ens_loss(
    model: Hard_OvR_Ens,
    x_batch: Array,
    y_batch: Array,
    β: int,
    train: bool = True,
    aggregation: str = 'mean',
) -> Callable:
    """Creates a loss function for training a Hard One-vs-Rest Ens."""
    def batch_loss(params, state):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            (loss, y, prod_ll, logZ), new_state = model.apply(
                {"params": params, **state}, x, y, train=train, β=β,
                mutable=list(state.keys()) if train else {},
            )

            return loss, new_state, y, prod_ll, logZ

        # broadcast over batch and aggregate
        agg = get_agg_fn(aggregation)
        loss_for_batch, new_state, ys, prod_lls, logZs = jax.vmap(
            loss_fn, out_axes=(0, None, 0, 0, 0), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return agg(loss_for_batch, axis=0), (new_state, ys, agg(prod_lls, axis=0), agg(logZs, axis=0))

    return batch_loss
