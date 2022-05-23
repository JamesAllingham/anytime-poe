from typing import Any, Optional, Union

import optax


ScalarOrSchedule = Union[float, optax.Schedule]

adam = optax.adam

adamw = optax.adamw

sgd = optax.sgd

def sgdw(
    learning_rate: ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    weight_decay: Optional[float] = None,
    accumulator_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    return optax.chain(
        (optax.trace(decay=momentum, nesterov=nesterov,
                     accumulator_dtype=accumulator_dtype)
        if momentum is not None else optax.identity()),
        (optax.add_decayed_weights(weight_decay)
        if weight_decay is not None else optax.identity()),
        _scale_by_learning_rate(learning_rate)
    )

def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return optax.scale_by_schedule(lambda count: m * learning_rate(count))
    return optax.scale(m * learning_rate)