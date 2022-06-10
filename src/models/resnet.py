from typing import Any, Callable, Optional
from functools import partial

import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers


Array = jnp.ndarray
ModuleDef = Any

class ResBlock(nn.Module):
    hidden_size: int
    dense: ModuleDef
    norm: ModuleDef
    act: Callable
    drop: Optional[ModuleDef] = None

    @nn.compact
    def __call__(self, x,):
        residual = x
        y = self.dense(self.hidden_size)(x)
        y = self.act(y)
        y = self.norm()(y)
        if self.drop is not None:
            y = self.drop()(y)

        return residual + y

pytorch_init = partial(initializers.variance_scaling, 1/3.0, "fan_in", "uniform")

class ResNet(nn.Module):
    depth: int
    hidden_size: int
    out_size: int
    kernel_init: Callable = pytorch_init()
    bias_init: Callable = pytorch_init(in_axis=-1, out_axis=-1)
    # ^ Not quite the same is what PyTorch does for biases since in_dim isn't always the same as out_dim
    p_drop: float = 0.

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        dense = partial(
            nn.Dense,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            epsilon=1e-6,
            momentum=0.9,
            # NOTE: 0.9 matches the PyTorch default of 0.1 as it is applied differently
            axis_name='batch',
        )
        drop = partial(
            nn.Dropout,
            rate=self.p_drop,
            deterministic=not train,
        ) if self.p_drop > 0. else None

        res_block = partial(
            ResBlock,
            dense=dense,
            norm=norm,
            act=nn.relu,
            drop=drop,
        )

        x = dense(self.hidden_size, name='input_layer')(x)

        for i in range(self.depth):
            x = res_block(self.hidden_size, name=f"layer_{i}")(x)

        return dense(self.out_size, name=f"output_layer")(x)