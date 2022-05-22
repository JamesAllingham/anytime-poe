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
    ens_preds = jnp.stack([net(x, train=train) for net in obj.nets], axis=0)  # (M, O * 2)
    M, _ = ens_preds.shape
    ens_preds = ens_preds.reshape(M, -1, 2)  # (M, O, 2)

    locs = ens_preds[:, :, 0]  # (M, O)

    if obj.noise == 'hetero':
        log_scales = ens_preds[:, :, 1]  # (M, O)
        scales = jnp.exp(log_scales)
    elif obj.noise == 'homo-per-ens':
        scales = jnp.exp(obj.logscale)  # (M, O)
    else:
        scales = jnp.repeat(jnp.exp(obj.logscale)[jnp.newaxis, :], M, axis=0)  # (M, O)

    probs = nn.softmax(obj.weights)[:, jnp.newaxis]

    return locs, scales, probs
