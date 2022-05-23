from typing import Callable, Mapping, Optional, Tuple, Union
from functools import partial

import wandb
from tqdm.auto import trange
import jax
from jax import random
from jax import numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from flax import struct
import flax.linen as nn
import optax
from chex import Array
from ml_collections import config_dict
from clu import parameter_overview

from src.data import NumpyLoader
import src.models as models
import src.utils.optim


PRNGKey = jnp.ndarray
ScalarOrSchedule = Union[float, optax.Schedule]


class TrainState(train_state.TrainState):
    """A Flax TrainState which also tracks model state (e.g. BatchNorm running averages) and schedules β."""
    model_state: FrozenDict
    β: float
    β_val_or_schedule: ScalarOrSchedule = struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            β=_get_β_for_step(self.step, self.β_val_or_schedule),
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, β_val_or_schedule, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            β_val_or_schedule=β_val_or_schedule,
            β=_get_β_for_step(0, β_val_or_schedule),
            **kwargs,
        )


# Helper which helps us deal with the fact that we can either specify a fixed β
# or a schedule for adjusting β. This is a pattern similar to the one used by
# optax for dealing with LRs either being specified as a constant or schedule.
def _get_β_for_step(step, β_val_or_schedule):
    if callable(β_val_or_schedule):
        return β_val_or_schedule(step)
    else:
        return β_val_or_schedule


def setup_training(
    config: config_dict.ConfigDict,
    rng: PRNGKey,
    init_x: Array,
    init_y: Union[float, int]
) -> Tuple[nn.Module, TrainState]:
    """Helper which returns the model object and the corresponding initialised train state for a given config.
    """
    model_cls = getattr(models, config.model_name)
    model = model_cls(**config.model.to_dict())

    init_rng, rng = random.split(rng)
    variables = model.init(init_rng, init_x, init_y)

    print(parameter_overview.get_parameter_overview(variables))
    # ^ This is really nice for summarising Jax models!

    model_state, params = variables.pop('params')
    del variables

    if config.get('lr_schedule_name', None):
        schedule = getattr(optax, config.lr_schedule_name)
        lr = schedule(
            init_value=config.learning_rate,
            **config.lr_schedule.to_dict()
        )
    else:
        lr = config.learning_rate

    optim = getattr(src.utils.optim, config.optim_name)
    optim = optax.inject_hyperparams(optim)
    # This ^ allows us to access the lr as opt_state.hyperparams['learning_rate'].

    if config.get('β_schedule', False):
        sigmoid = lambda x: 1 / (1 + jnp.exp(x))
        add = config.β_schedule.end - config.β_schedule.start
        half_steps = config.β_schedule.steps/2
        β = lambda step: config.β_schedule.start + add * sigmoid((-step + half_steps)/(config.β_schedule.steps/10))
    else:
        β = None

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optim(learning_rate=lr, **config.optim.to_dict()),
        model_state=model_state,
        β_val_or_schedule=β,
    )

    return model, state


def train_loop(
    model: nn.Module,
    state: TrainState,
    config: config_dict.ConfigDict,
    rng: PRNGKey,
    make_loss_fn: Callable,
    make_eval_fn: Callable,
    train_loader: NumpyLoader,
    val_loader: NumpyLoader,
    test_loader: Optional[NumpyLoader] = None,
    wandb_kwargs: Optional[Mapping] = None,
) -> TrainState:
    """Runs the training loop!
    """
    wandb_kwargs = {
        'project': 'anytime-poe',
        'entity': 'jamesallingham',
        'notes': '',
        # 'mode': 'disabled',
        'config': config.to_dict()
    } | (wandb_kwargs or {})
    # ^ here wandb_kwargs (i.e. whatever the user specifies) takes priority.

    with wandb.init(**wandb_kwargs) as run:
        @jax.jit
        def train_step(state, x_batch, y_batch, rng):
            kwargs = {'β': state.β} if state.β is not None else {}
            loss_fn = make_loss_fn(model, x_batch, y_batch, train=True, aggregation='sum', **kwargs)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            (nll, model_state), grads = grad_fn(
                state.params, state.model_state, #rng,
            )

            return state.apply_gradients(grads=grads, model_state=model_state), nll

        @jax.jit
        def eval_step(state, x_batch, y_batch, rng):
            kwargs = {'β': state.β} if state.β is not None else {}
            eval_fn = make_eval_fn(model, x_batch, y_batch, train=False, aggregation='sum', **kwargs)

            nll, _ = eval_fn(
                state.params, state.model_state, #rng,
            )

            return nll


        train_losses = []
        val_losses = []
        epochs = trange(1, config.epochs + 1)
        for epoch in epochs:
            batch_losses = []
            for (x_batch, y_batch) in train_loader:
                rng, batch_rng = random.split(rng)
                state, nll = train_step(state, x_batch, y_batch, batch_rng)
                batch_losses.append(nll)

            train_losses.append(jnp.sum(jnp.array(batch_losses)) / len(train_loader.dataset))

            batch_losses = []
            for (x_batch, y_batch) in val_loader:
                rng, eval_rng = random.split(rng)
                nll = eval_step(state, x_batch, y_batch, eval_rng)
                batch_losses.append(nll)

            val_losses.append(jnp.sum(jnp.array(batch_losses)) / len(val_loader.dataset))

            learning_rate = state.opt_state.hyperparams['learning_rate']
            metrics_str = (f'train loss: {train_losses[-1]:7.5f}, val_loss: {val_losses[-1]:7.5f}' +
                           (f', β: {state.β:3.1f}' if state.β is not None else '') +
                           f', lr: {learning_rate:7.5f}')
            epochs.set_postfix_str(metrics_str)
            print(f'epoch: {epoch:3} - {metrics_str}')

            metrics = {
                'epoch': epoch,
                'train/loss': train_losses[-1],
                'val/loss': val_losses[-1],
                'β': state.β,
                'learning_rate': learning_rate,
            }
            run.log(metrics)

            rng, test_rng = random.split(rng)
            if val_losses[-1] <= min(val_losses):
                print("Best val_loss")
                # TODO: add model saving.

                run.summary['best_epoch'] = epoch
                run.summary['best_val_loss'] = val_losses[-1]

                if test_loader is not None:
                    batch_losses = []
                    for (x_batch, y_batch) in (test_loader):
                        eval_rng, test_rng = random.split(test_rng)
                        nll = eval_step(state, x_batch, y_batch, eval_rng)
                        batch_losses.append(nll)

                    test_loss = jnp.sum(jnp.array(batch_losses)) / len(test_loader.dataset)
                    run.summary['test/loss'] = test_loss

    return state
