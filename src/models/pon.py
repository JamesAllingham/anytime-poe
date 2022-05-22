from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array
import distrax

from src.models.common import raise_if_not_in_list, NOISE_TYPES, get_locs_scales_probs


class PoN_Ens(nn.Module):
    """Ens trained as a Product of Normals."""
    size: int
    make_net: Callable[[], nn.Module]
    weights_init: Callable = initializers.ones
    logscale_init: Callable = initializers.zeros
    noise: str = 'homo'
    learn_weights: bool = False

    def setup(self):
        raise_if_not_in_list(self.noise, NOISE_TYPES, 'self.noise')

        self.nets = [self.make_net() for _ in range(self.size)]
        self.weights = self.param(
            'weights',
            self.weights_init,
            (self.size,)
        )
        if self.noise != 'hetero':
            self.logscale = self.param(
                'logscale',
                self.logscale_init,
                (self.nets[0].out_size // 2,) if self.noise == 'homo' else (self.size, self.net.out_size // 2,)
            )

    def __call__(
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
) -> Callable:
    """Creates a loss function for training a PoE Ens."""
    def batch_loss(params, state):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            (loc, scale), new_state = model.apply(
                {"params": params, **state}, x, train=train,
                mutable=list(state.keys()) if train else {},
            )

            nll = -1 * distrax.Normal(loc, scale).log_prob(y)

            sq_err = (loc - y)**2

            return nll[0], sq_err[0], new_state

        # broadcast over batch and take mean
        loss_for_batch, errors_for_batch, new_state = jax.vmap(
            loss_fn, out_axes=(0, 0, None), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return loss_for_batch.mean(axis=0), (errors_for_batch.mean(axis=0), new_state)

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