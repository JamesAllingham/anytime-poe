from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array
import distrax

from src.models.common import raise_if_not_in_list, NOISE_TYPES, get_locs_scales_probs


class Reg_Ens(nn.Module):
    """A standard regression ensemble."""
    size: int
    make_net: Callable[[], nn.Module]
    weights_init: Callable = initializers.ones
    logscale_init: Callable = initializers.zeros
    noise: str = 'homo'
    learn_weights: bool = False

    def setup(self):
        raise_if_not_in_list(self.noise, NOISE_TYPES, 'self.noise')

        self.nets = [self.make_net() for _ in range(self.size)]
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

        def nll(y, loc, scale):
            return  -1 * distrax.Normal(loc, scale).log_prob(y)

        nlls = jax.vmap(nll, in_axes=(None, 0, 0))(y, locs, scales)
        loss = (nlls * probs).sum(axis=0)

        return loss

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
    ):
        locs, scales, probs = get_locs_scales_probs(self, x, train)

        loc = (locs * probs).sum(axis=0)
        σ2 = ((locs**2 + scales**2) * probs).sum(axis=0) - loc**2
        scale = jnp.sqrt(σ2)

        if return_ens_preds:
            return (loc, scale), (locs, scales)
        else:
            return (loc, scale)


def make_Reg_Ens_loss(
    model: Reg_Ens,
    x_batch: Array,
    y_batch: Array,
    train: bool = True,
) -> Callable:
    """Creates a loss function for training a std Ens."""
    def batch_loss(params, state):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            loss, new_state = model.apply(
                {"params": params, **state}, x, y, train=train,
                mutable=list(state.keys()) if train else {},
            )

            return loss, new_state

        # broadcast over batch and take mean
        loss_for_batch, new_state = jax.vmap(
            loss_fn, out_axes=(0, None), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return loss_for_batch.mean(axis=0), new_state

    return batch_loss
