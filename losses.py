# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from utils import batch_mul, get_div_fn, get_value_div_fn
import functools
import numpy as np
from bound_likelihood import get_likelihood_offset_fn


def get_optimizer(config, beta2=0.999):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = flax.optim.Adam(beta1=config.optim.beta1, beta2=beta2, eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config, deq_score_joint=False):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(state,
                  grad,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    lr = state.lr
    if warmup > 0:
      lr = lr * jnp.minimum(state.step / warmup, 1.0)
    if grad_clip >= 0:
      # Compute global gradient norm
      grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
      # Clip gradient
      clipped_grad = jax.tree_map(
        lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
    else:  # disabling gradient clipping if grad_clip < 0
      clipped_grad = grad
    return state.optimizer.apply_gradient(clipped_grad, learning_rate=lr)

  def optimize_deq_score_fn(state,
                            grad,
                            warmup=config.optim.warmup,
                            grad_clip=config.optim.grad_clip):
    lr = state.lr
    if warmup > 0:
      lr = lr * jnp.minimum(state.step / warmup, 1.0)
    if grad_clip >= 0:
      # Compute global gradient norm
      grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
      # Clip gradient
      clipped_grad = jax.tree_map(
        lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
    else:  # disabling gradient clipping if grad_clip < 0
      clipped_grad = grad
    return (state.deq_optimizer.apply_gradient(clipped_grad['deq'], learning_rate=lr),
            state.score_optimizer.apply_gradient(clipped_grad['score'], learning_rate=lr))

  if deq_score_joint:
    return optimize_deq_score_fn
  else:
    return optimize_fn


def get_score_t(score_fn):
  def score_t(x, t, rng):
    tangent = jnp.ones_like(t)
    return jax.jvp(lambda time: score_fn(x, time, rng=rng), (t,), (tangent,))

  return score_t


def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True,
                    importance_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    importance_weighting: If `True`, use importance weighting to reduce the variance of likelihood weighting.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """

    score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)
    data = batch['image']

    rng, step_rng = random.split(rng)
    if likelihood_weighting and importance_weighting:
      t = sde.sample_importance_weighted_time_for_likelihood(step_rng, (data.shape[0],), eps=eps)
    else:
      t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)

    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, data.shape)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)
    rng, step_rng = random.split(rng)
    score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

    if likelihood_weighting:
      if importance_weighting:
        losses = jnp.square(batch_mul(score, std) + z)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
      else:
        g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
        losses = jnp.square(score + batch_mul(z, 1. / std))
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

    else:
      losses = jnp.square(batch_mul(score, std) + z)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)

    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_step_fn(sde, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False,
                importance_weighting=False, smallest_time=1e-5):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting,
                              importance_weighting=importance_weighting,
                              eps=smallest_time)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    if train:
      params = state.optimizer.target
      states = state.model_state
      (loss, new_model_state), grad = grad_fn(step_rng, params, states, batch)

      grad = jax.lax.pmean(grad, axis_name='batch')
      new_optimizer = optimize_fn(state, grad)
      new_params_ema = jax.tree_multimap(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_optimizer.target
      )
      step = state.step + 1
      new_state = state.replace(
        step=step,
        optimizer=new_optimizer,
        model_state=new_model_state,
        params_ema=new_params_ema
      )
    else:
      loss, _ = loss_fn(step_rng, state.params_ema, state.model_state, batch)
      new_state = state

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, loss

  return step_fn


def get_dequantization_loss_fn(sde, score_fn, deq_model, scaler, inverse_scaler,
                               train=True, importance_weighting=True, eps=1e-5,
                               eps_offset=True):
  def div_drift_fn(x, t, eps):
    div_fn = get_div_fn(lambda x, t: sde.sde(x, t)[0])
    return div_fn(x, t, eps)

  def loss_fn(rng, params, batch):
    dequantizer = mutils.get_dequantizer(deq_model, params, train=train)

    data = batch['image']
    shape = data.shape
    rng, step_rng = random.split(rng)
    u = random.normal(step_rng, shape)
    if train:
      rng, step_rng = random.split(rng)
      deq_noise, sldj = dequantizer(u, inverse_scaler(data), rng=step_rng)
    else:
      deq_noise, sldj = dequantizer(u, inverse_scaler(data))

    data = scaler((inverse_scaler(data) * 255. + deq_noise) / 256.)

    mean, std = sde.marginal_prob(data, jnp.ones((shape[0],)) * sde.T)
    rng, step_rng = jax.random.split(rng)
    z = jax.random.normal(step_rng, shape)
    neg_prior_logp = -sde.prior_logp(mean + batch_mul(std, z))

    rng, step_rng = random.split(rng)
    if importance_weighting:
      t = sde.sample_importance_weighted_time_for_likelihood(step_rng, (shape[0],), eps=eps)
      Z = sde.likelihood_importance_cum_weight(sde.T, eps=eps)
    else:
      t = random.uniform(step_rng, (shape[0],), minval=eps, maxval=sde.T)

    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, shape)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)

    score = score_fn(perturbed_data, t)
    if importance_weighting:
      losses = jnp.square(batch_mul(score, std) + z)
      losses = jnp.sum(losses.reshape((losses.shape[0], -1)), axis=-1)
      grad_norm = jnp.square(z).reshape((z.shape[0], -1)).sum(axis=-1)
      losses = (losses - grad_norm) * Z
    else:
      g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
      losses = jnp.square(score + batch_mul(z, 1. / std))
      losses = jnp.sum(losses.reshape((losses.shape[0], -1)), axis=-1) * g2
      grad_norm = jnp.square(z).reshape((z.shape[0], -1)).sum(axis=-1)
      grad_norm = grad_norm * g2 / (std ** 2)
      losses = losses - grad_norm

    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, shape)
    rng, step_rng = random.split(rng)
    t = random.uniform(step_rng, (shape[0],), minval=eps, maxval=sde.T)
    mean, std = sde.marginal_prob(data, t)
    noisy_data = mean + batch_mul(std, z)
    rng, step_rng = random.split(rng)
    epsilon = random.rademacher(step_rng, shape, dtype=jnp.float32)
    drift_div = div_drift_fn(noisy_data, t, epsilon)

    losses = neg_prior_logp + 0.5 * (losses - 2 * drift_div) - sldj
    if eps_offset:
      offset_fn = get_likelihood_offset_fn(sde, score_fn, eps)
      rng, step_rng = random.split(rng)
      losses = losses + offset_fn(step_rng, data)

    dim = np.prod(shape[1:])
    bpd = losses / np.log(2.)
    bpd = bpd / dim
    offset = jnp.log2(jax.grad(inverse_scaler)(0.)) + 8.
    bpd += offset
    bpd = bpd.mean()

    loss = jnp.mean(losses)

    return loss, bpd

  return loss_fn


def get_dequantizer_step_fn(sde, score_fn, deq_model, scaler, inverse_scaler,
                            train, optimize_fn=None, importance_weighting=False, smallest_time=1e-5,
                            eps_offset=True):
  loss_fn = get_dequantization_loss_fn(sde, score_fn, deq_model, scaler, inverse_scaler, train,
                                       importance_weighting=importance_weighting, eps=smallest_time,
                                       eps_offset=eps_offset)

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    if train:
      params = state.optimizer.target
      (loss, bpd), grad = grad_fn(step_rng, params, batch)

      grad = jax.lax.pmean(grad, axis_name='batch')
      new_optimizer = optimize_fn(state, grad)
      new_params_ema = jax.tree_multimap(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_optimizer.target
      )
      step = state.step + 1
      new_state = state.replace(
        step=step,
        optimizer=new_optimizer,
        params_ema=new_params_ema
      )
    else:
      loss, bpd = loss_fn(step_rng, state.params_ema, batch)
      new_state = state

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, (loss, bpd)

  return step_fn
