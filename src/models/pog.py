from typing import Any, Callable, Mapping
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array
import matplotlib.pyplot as plt

from src.models.common import raise_if_not_in_list, NOISE_TYPES, get_locs_scales_probs, get_agg_fn
from src.models.resnet import ResNet
from src.models.pon import normal_prod


KwArgs = Mapping[str, Any]


def gnd_ll(y, loc, scale, β):
    per_dim_lls = jnp.log(β) - jnp.log(2*scale) - jax.scipy.special.gammaln(1/β) - (jnp.abs(y - loc)/scale)**β
    return jnp.sum(per_dim_lls, axis=0, keepdims=True)


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
                (self.net['out_size'],) if self.noise == 'homo' else (self.size, self.net['out_size'],)
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
            prod_lls = jax.vmap(gnd_ll, in_axes=(None, 0, 0, None))(y, locs, scales, β)
            return jnp.sum(probs * prod_lls)

        dy = 0.001
        ys = jnp.arange(-10, 10 + dy, dy)
        ps = jnp.exp(jax.vmap(product_logprob)(ys))
        Z = jnp.trapz(ps, ys)

        log_prob = product_logprob(y)
        nll = -(log_prob - jnp.log(Z + 1e-36))

        return nll

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
    ) -> Array:
        locs, scales, _ = get_locs_scales_probs(self, x, train)

        loc, scale = calculate_pog_loc_scale(locs, scales)

        if return_ens_preds:
            return (loc, scale), (locs, scales)
        else:
            return (loc, scale)


def calculate_pog_loc_scale(locs, scales):
    mins = locs - scales
    maxs = locs + scales

    max = jnp.min(maxs, axis=0)
    min = jnp.max(mins, axis=0)

    # max = max.at[min > max].set(0)
    # min = min.at[min > max].set(0)
    max, min = jnp.where(min > max, jnp.nan, max), jnp.where(min > max, jnp.nan, min)

    scale = (max - min)/2
    loc = max - scale

    return loc, scale


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


def make_PoG_Ens_plots(
    pog_model, pog_state, pog_tloss, pog_vloss, X_train, y_train,
    ):
    pog_params, pog_model_state = pog_state.params, pog_state.model_state

    n_plots = 2
    fig, axs = plt.subplots(1, n_plots, figsize=(7.5 * n_plots, 6))

    xs = jnp.linspace(-3, 3, num=501)

    # pog preds
    pred_fun = partial(
        pog_model.apply,
        {"params": pog_params, **pog_model_state},
        train=False, return_ens_preds=True,
        method=pog_model.pred
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

    norm_scales = (scales*2)**0.5
    norm_loc, norm_scale = normal_prod(locs, norm_scales, nn.softmax(pog_params['weights'])[:, jnp.newaxis])
    axs[0].plot(xs, norm_loc, '--', c='C2', alpha=0.5)
    axs[0].fill_between(xs, norm_loc - norm_scale, norm_loc + norm_scale, color='C2', alpha=0.1)

    axs[0].plot(xs, loc, c='C1')
    axs[0].fill_between(xs, loc - scale, loc + scale, color='C1', alpha=0.4)


    axs[0].set_title(f"PoG - train loss: {pog_tloss:.6f}, val loss: {pog_vloss:.6f}")
    axs[0].set_ylim(-2.5, 2.5)
    axs[0].set_xlim(-3, 3)


    # plot locs and scales for each member
    axs[1].scatter(X_train, y_train, c='C0')
    for i in range(size):
        axs[1].plot(xs, locs[i], alpha=0.5)
        axs[1].fill_between(xs, locs[i] - scales[i], locs[i] + scales[i], alpha=0.1)
    axs[1].set_title(f"PoG Members")
    axs[1].set_ylim(-2.5, 2.5)
    axs[1].set_xlim(-3, 3)

    plt.show()

    return fig
