from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array
import distrax

from src.models.common import raise_if_not_in_list, NOISE_TYPES, get_locs_scales_probs, get_agg_fn
from src.models.resnet import ResNet


KwArgs = Mapping[str, Any]


class PoN_Ens(nn.Module):
    """Ens trained as a Product of Normals."""
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
    ) -> Array:
        locs, scales, probs = get_locs_scales_probs(self, x, train)

        loc, scale = normal_prod(locs, scales, probs)

        nll = -distrax.Normal(loc, scale).log_prob(y)

        return nll

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
    ) -> Array:
        locs, scales, probs = get_locs_scales_probs(self, x, train)

        loc, scale = normal_prod(locs, scales, probs)

        if return_ens_preds:
            return (loc, scale), (locs, scales)
        else:
            return (loc, scale)


def make_PoN_Ens_loss(
    model: PoN_Ens,
    x_batch: Array,
    y_batch: Array,
    train: bool = True,
    aggregation: str = 'mean',
) -> Callable:
    """Creates a loss function for training a PoE Ens."""
    def batch_loss(params, state):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            nll, new_state = model.apply(
                {"params": params, **state}, x, y, train=train,
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


def normal_prod(locs, scales, probs=None):
    if probs == None:
        probs = jnp.ones_like(locs)

    scales2 = scales ** 2
    θ_1 = ((locs / scales2) * probs).sum(axis=0)
    θ_2 = ((-1 / (2 * scales2)) * probs).sum(axis=0)

    σ2 = -1 / (2 * θ_2)
    scale = jnp.sqrt(σ2)
    loc = θ_1 * σ2

    return loc, scale
