from typing import Any, Callable, Mapping, Optional
from functools import partial
from attr import mutable

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from chex import Array, assert_rank, assert_equal_shape
import matplotlib.pyplot as plt

from src.models.common import get_agg_fn
from src.models.resnet import ResNet


KwArgs = Mapping[str, Any]


def hardened_ovr_ll(y_1hot, logits, T):
    assert_rank(T, 0)
    assert_rank(y_1hot, 1)
    assert_equal_shape([y_1hot, logits])

    σ = nn.sigmoid(T * logits).clip(1e-6, 1 - 1e-6)
    res = jnp.sum(y_1hot * jnp.log(σ) + (1 - y_1hot) * jnp.log(1 - σ), axis=0)
    return res


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
            y_1hot = jax.nn.one_hot(y, n_classes)  # TODO: this would not work for pixelwise classification
            lls = jax.vmap(hardened_ovr_ll, in_axes=(None, 0, None))(y_1hot, ens_logits, β)
            res = jnp.sum(probs * lls, axis=0)
            return res

        ys = jnp.arange(n_classes)
        Z = jnp.sum(jnp.exp(jax.vmap(product_logprob)(ys)), axis=0)

        prod_ll = product_logprob(y)
        nll = -(prod_ll - jnp.log(Z + 1e-36))

        prod_preds = nn.sigmoid(β * ens_logits).prod(axis=0)
        pred = prod_preds.argmax(axis=0)
        err = y != pred

        return nll, err

    def pred(
        self,
        x: Array,
        train: bool = False,
        return_ens_preds = False,
        hard_pred = False,
        β: int = 1,
    ) -> Array:
        ens_logits = jnp.stack([net(x, train=train) for net in self.nets], axis=0)  # (M, O)
        ens_preds = nn.sigmoid(β * ens_logits)
        if hard_pred:
            ens_preds = jnp.round(ens_preds)

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
    def batch_loss(params, state, rng):
        # define loss func for 1 example
        def loss_fn(params, x, y):
            (loss, err), new_state = model.apply(
                {"params": params, **state}, x, y, train=train, β=β,
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


def make_Hard_OvR_Ens_toy_plots(
    hard_ovr_model, hard_ovr_state, hard_ovr_tloss, hard_ovr_vloss, X_train, y_train,
):
    hard_ovr_params, hard_ovr_model_state = hard_ovr_state.params, hard_ovr_state.model_state

    n_plots = 1
    fig, axs = plt.subplots(1, n_plots, figsize=(7.5 * n_plots, 6))

    X_train, y_train = jnp.array(X_train), jnp.array(y_train)
    n_class = int(y_train.max()) + 1

    # hard_ovr preds
    h = .05  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X_train[:, 0].min() * 1.25, X_train[:, 0].max() * 1.25
    y_min, y_max = X_train[:, 1].min() * 1.25, X_train[:, 1].max() * 1.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    xs = np.c_[xx.ravel(), yy.ravel()]

    pred_fun = partial(
        hard_ovr_model.apply,
        {"params": hard_ovr_params, **hard_ovr_model_state},
        train=False, return_ens_preds=True, β=hard_ovr_state.β,
        method=hard_ovr_model.pred
    )
    preds, _ = jax.vmap(
        pred_fun, out_axes=(0, 1), in_axes=(0,), axis_name="batch"
    )(xs)

    # size = preds.shape[0]

    colormaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']
    for i in range(n_class):
        axs.pcolormesh(xx, yy, preds[:, i].reshape(xx.shape), alpha=0.25, cmap=colormaps[i])

    # for i in range(depth + 1):
    #     axs.contour(xx, yy, ens_preds[:, i, 0].reshape(xx.shape), cmap=plt.cm.gray, levels=[.5], alpha=0.3)

    markers = ['o', 'v', 's', 'P', 'X']
    for i in range(n_class):
        idxs = (y_train == i)
        axs.plot(X_train[idxs[:, 0], 0], X_train[idxs[:, 0], 1], markers[0], c=f'C{i}', alpha=1, ms=1)

    plt.show()

    return fig


def make_Hard_OvR_Ens_MNIST_plots(
    hard_ovr_model, hard_ovr_state, hard_ovr_tloss, hard_ovr_vloss, X_train, y_train, X_val, y_val,
):
    hard_ovr_params, hard_ovr_model_state = hard_ovr_state.params, hard_ovr_state.model_state

    n_plots = 8
    fig, axs = plt.subplots(1, n_plots, figsize=(2.5 * n_plots, 2), layout='tight')
    fig.patch.set_alpha(1.)

    X_train, y_train = jnp.array(X_train), jnp.array(y_train)
    X_val, y_val = jnp.array(X_val), jnp.array(y_val)

    # hard_ovr preds
    xs = X_val
    ys = y_val

    pred_fun = partial(
        hard_ovr_model.apply,
        {"params": hard_ovr_params, **hard_ovr_model_state},
        train=False, return_ens_preds=False, β=hard_ovr_state.β,
        method=hard_ovr_model.pred
    )
    preds = jax.vmap(
        pred_fun, axis_name="batch"
    )(xs)


    pred_fun = partial(
        hard_ovr_model.apply,
        {"params": hard_ovr_params, **hard_ovr_model_state},
        train=False, β=hard_ovr_state.β,
    )
    lls, _ = jax.vmap(
        pred_fun, axis_name="batch", out_axes=(0, 0),
    )(xs, ys)

    # get idxs of NaN values in lls
    # idxs = jnp.where(jnp.isnan(lls))[0]
    # print(f'{len(idxs)}/{len(lls)} NaN values in lls')
    # print(f'{idxs}')
    # for pred, y in zip(preds[idxs], ys[idxs]):
    #     print(f'pred: {pred}')
    #     print(f'y: {y}')
    #     print(hardened_ovr_ll(jax.nn.one_hot(y, 10), jnp.log(pred/(1 - pred)), hard_ovr_state.β))

    for idx in range(n_plots):
        axs[idx].imshow(xs[idx].reshape(28, 28), cmap='gray')
        axs[idx].set_title(f'{jnp.argmax(preds[idx])} – {jnp.max(preds[idx]):.4f} ({ys[idx]} – {preds[idx][ys[idx]]:.4f})', fontsize=12)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])

    plt.show()

    return fig
