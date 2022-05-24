from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
from chex import Array

NOISE_TYPES = [
    'homo',
    'per-ens-homo',
    'hetero',
]

def raise_if_not_in_list(val, valid_options, varname):
    if val not in valid_options:
       msg = f'`{varname}` should be one of `{valid_options}` but was `{val}` instead.'
       raise RuntimeError(msg)


def get_locs_scales_probs(
    obj,
    x: Array,
    train: bool = False,
):
    ens_preds = jnp.stack([net(x, train=train) for net in obj.nets], axis=0)  # (M, O * 2) or (M, O)
    M, _ = ens_preds.shape

    if obj.noise == 'hetero':
        ens_preds = ens_preds.reshape(M, -1, 2)  # (M, O, 2)
        locs = ens_preds[:, :, 0]  # (M, O)
        log_scales = ens_preds[:, :, 1]  # (M, O)
        scales = jnp.exp(log_scales)
    elif obj.noise == 'homo-per-ens':
        locs = ens_preds
        scales = jnp.exp(obj.logscale)  # (M, O)
    else:
        locs = ens_preds
        scales = jnp.repeat(jnp.exp(obj.logscale)[jnp.newaxis, :], M, axis=0)  # (M, O)

    probs = nn.softmax(obj.weights)[:, jnp.newaxis]

    return locs, scales, probs


def get_agg_fn(agg: str) -> Callable:
    raise_if_not_in_list(agg, ['mean', 'sum'], 'aggregation')

    if agg == 'mean':
        return jnp.mean
    else:
        return jnp.sum
