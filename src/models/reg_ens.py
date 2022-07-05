from functools import partial
from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array
import distrax
import matplotlib.pyplot as plt

from src.models.common import raise_if_not_in_list, NOISE_TYPES, get_locs_scales_probs, get_agg_fn
from src.models.resnet import ResNet


KwArgs = Mapping[str, Any]


class Reg_Ens(nn.Module):
    """A standard regression ensemble."""
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
                (self.net['out_size'],) if self.noise == 'homo' else (self.size, self.net['out_size'],)
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
        loss = (nlls).sum(axis=0)[0]
        # TODO: include weights, but don't decrease effective LR to each ens member?

        err = jnp.mean((locs.mean(axis=0) - y)**2)

        return loss, err

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
    ):
        locs, scales, probs = get_locs_scales_probs(self, x, train)

        loc, scale = calculate_ens_loc_scale(locs, scales, probs)

        if return_ens_preds:
            return (loc, scale), (locs, scales)
        else:
            return (loc, scale)


def calculate_ens_loc_scale(locs, scales, probs):
    loc = (locs * probs).sum(axis=0)
    σ2 = ((locs**2 + scales**2) * probs).sum(axis=0) - loc**2
    scale = jnp.sqrt(σ2)

    return loc, scale


def make_Reg_Ens_loss(
    model: Reg_Ens,
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
            )

            return loss, new_state, err

        # broadcast over batch and aggregate
        agg = get_agg_fn(aggregation)
        loss_for_batch, new_state, err_for_batch = jax.vmap(
            loss_fn, out_axes=(0, None, 0), in_axes=(None, 0, 0), axis_name="batch"
        )(params, x_batch, y_batch)
        return agg(loss_for_batch, axis=0), (new_state, agg(err_for_batch, axis=0))

    return batch_loss


def make_Reg_Ens_plots(
    ens_model, ens_state, ens_tloss, ens_vloss, X_train, y_train, X_val, y_val,
    ):
    ens_params, ens_model_state = ens_state.params, ens_state.model_state

    n_plots = 2
    fig, axs = plt.subplots(1, n_plots, figsize=(7.5 * n_plots, 6))

    xs = jnp.linspace(-3, 3, num=501)

    # ens preds
    pred_fun = partial(
        ens_model.apply,
        {"params": ens_params, **ens_model_state},
        train=False, return_ens_preds=True,
        method=ens_model.pred
    )
    (loc, scale), (locs, scales) = jax.vmap(
        pred_fun, out_axes=(0, 1), in_axes=(0,), axis_name="batch"
    )(xs.reshape(-1, 1))

    size = locs.shape[0]

    loc = loc[:, 0]
    scale = scale[:, 0]
    locs = locs[:, :, 0]
    scales = scales[:, :, 0]

    axs[0].scatter(X_train, y_train, c='C0')
    for i in range(size):
        axs[0].plot(xs, locs[i], c='k', alpha=0.25)


    axs[0].plot(xs, loc, c='C1')
    axs[0].fill_between(xs, loc - scale, loc + scale, color='C1', alpha=0.4)

    axs[0].set_title(f"Reg Ens - train loss: {ens_tloss:.6f}, val loss: {ens_vloss:.6f}")
    axs[0].set_ylim(-2.5, 2.5)
    axs[0].set_xlim(-3, 3)


    # plot locs and scales for each member
    axs[1].scatter(X_train, y_train, c='C0')
    for i in range(size):
        axs[1].plot(xs, locs[i], alpha=0.5)
        axs[1].fill_between(xs, locs[i] - scales[i], locs[i] + scales[i], alpha=0.1)
    axs[1].set_title(f"Ens Members")
    axs[1].set_ylim(-2.5, 2.5)
    axs[1].set_xlim(-3, 3)

    plt.show()

    return fig