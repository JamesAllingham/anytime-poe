from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array

from src.models.common import raise_if_not_in_list, NOISE_TYPES, get_locs_scales_probs, get_agg_fn
from src.models.resnet import ResNet


KwArgs = Mapping[str, Any]


def gnd_ll(y, loc, scale, β):
    return jnp.log(β) - jnp.log(2*scale) - jax.scipy.special.gammaln(1/β) - (jnp.abs(y - loc)/scale)**β


class PoG_Ens(nn.Module):
    """Ens trained as a Product of GNDs."""
    size: int
    net: KwArgs
    weights_init: Callable = initializers.ones
    logscale_init: Callable = initializers.zeros
    noise: str = 'homo'
    learn_weights: bool = False

    def setup(self):
        raise_if_not_in_list(self.noise, NOISE_TYPES, 'self.noise')

        self.nets = [ResNet(**self.net) for _ in range(self.size)]
        weights = self.param(
            'weights',
            self.weights_init,
            (self.size,)
        )
        self.weights = weights if self.learn_weights else jax.lax.stop_gradient(weights)
        if self.noise != 'hetero':
            self.logscale = self.param(
                'logscale',
                self.logscale_init,
                (self.nets[0].out_size // 2,) if self.noise == 'homo' else (self.size, self.net.out_size // 2,)
            )

    def __call__(
        self,
        x: Array,
        y: int,
        train: bool = False,
        β: int = 2,
    ) -> Array:
        locs, scales, probs = get_locs_scales_probs(self, x, train)

        def product_logprob(y):
            return jnp.sum(probs * jax.vmap(gnd_ll, in_axes=(None, 0, 0, None))(y, locs, scales, β))

        dy = 0.001
        ys = jnp.arange(-10, 10 + dy, dy)
        ps = jnp.exp(jax.vmap(product_logprob)(ys))
        Z = jnp.trapz(ps, ys)

        nll = -(product_logprob(y) - jnp.log(Z + 1e-36))

        return nll

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
    ) -> Array:
        locs, scales, _ = get_locs_scales_probs(self, x, train)

        mins = locs - scales
        maxs = locs + scales

        max = jnp.min(maxs, axis=0)
        min = jnp.max(mins, axis=0)

        # max = max.at[min > max].set(0)
        # min = min.at[min > max].set(0)
        max, min = jnp.where(min > max, jnp.nan, max), jnp.where(min > max, jnp.nan, min)

        scale = (max - min)/2
        loc = max - scale

        if return_ens_preds:
            return (loc, scale), (locs, scales)
        else:
            return (loc, scale)


def make_PoG_Ens_loss(
    model: PoG_Ens,
    x_batch: Array,
    y_batch: Array,
    β: int,
    train: bool = True,
    # ^ controls how much our GND looks like a Guassian (β=2) or Uniform (β->inf)
    # should be taken from 2 to ??? duringn the process of training
    aggregation: str = 'mean',
) -> Callable:
    """Creates a loss function for training a PoE DUN."""
    def batch_loss(params, state):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            nll, new_state = model.apply(
                {"params": params, **state}, x, y, train=train, β=β,
                mutable=list(state.keys()) if train else {},
            )

            return nll, new_state

        # broadcast over batch and aggregate
        agg = get_agg_fn(aggregation)
        loss_for_batch, new_state = jax.vmap(
            loss_fn, out_axes=(0, None), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return agg(loss_for_batch, axis=0), new_state

    return batch_loss
