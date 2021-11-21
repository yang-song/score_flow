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

"""All functions and modules related to model definition.
"""
from typing import Any

import flax
import functools
import jax.numpy as jnp

import datasets
import sde_lib
import jax
import numpy as np
from flax.training import checkpoints
from utils import batch_mul


# The dataclass that stores all training states
@flax.struct.dataclass
class State:
  step: int
  optimizer: flax.optim.Optimizer
  lr: float
  model_state: Any
  ema_rate: float
  params_ema: Any
  rng: Any


@flax.struct.dataclass
class DeqState:
  step: int
  optimizer: flax.optim.Optimizer
  lr: float
  ema_rate: float
  params_ema: Any
  ema_train_bpd: float
  ema_eval_bpd: float
  rng: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """

  if config.training.sde.lower() == 'linearvesde':
    sigmas = jnp.sqrt(jnp.linspace(config.model.sigma_max ** 2, config.model.sigma_min ** 2,
                                   config.model.num_scales))
  else:
    sigmas = jnp.exp(
      jnp.linspace(
        jnp.log(config.model.sigma_max), jnp.log(config.model.sigma_min),
        config.model.num_scales))
  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def init_model(rng, config, data=None, label=None):
  """ Initialize a `flax.linen.Module` model. """
  model_name = config.model.name
  model_def = functools.partial(get_model(model_name), config=config)
  input_shape = (config.training.batch_size // jax.local_device_count(),
                 config.data.image_size, config.data.image_size, config.data.num_channels)
  label_shape = input_shape[:1]
  if data is None:
    init_input = jnp.zeros(input_shape)
  else:
    init_input = data
  if label is None:
    init_label = jnp.zeros(label_shape, dtype=jnp.int32)
  else:
    init_label = label
  params_rng, dropout_rng = jax.random.split(rng)
  model = model_def()
  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, init_input, init_label)
  # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
  init_model_state, initial_params = variables.pop('params')
  return model, init_model_state, initial_params


def data_dependent_init_of_dequantizer(rng, config, init_data):
  if config.data.dataset == 'ImageNet':
    if config.data.image_size == 32:
      from .flowpp import dequantization_imagenet32
      model = dequantization_imagenet32.Dequantization()
    elif config.data.image_size == 64:
      from .flowpp import dequantization_imagenet64
      model = dequantization_imagenet64.Dequantization()
  elif config.data.dataset == 'CIFAR10':
    from .flowpp import dequantization_cifar10
    model = dequantization_cifar10.Dequantization()

  rng, step_rng = jax.random.split(rng)
  u = jax.random.normal(step_rng, init_data.shape)

  @functools.partial(jax.pmap, axis_name='batch')
  def init_func(params_rng, dropout_rng, eps, data):
    return model.init({'params': params_rng, 'dropout': dropout_rng}, eps, data, inverse=False, train=False)

  rng, *params_rng = jax.random.split(rng, jax.local_device_count() + 1)
  params_rng = jnp.asarray(params_rng)
  rng, *dropout_rng = jax.random.split(rng, jax.local_device_count() + 1)
  dropout_rng = jnp.asarray(dropout_rng)
  variables = flax.jax_utils.unreplicate(init_func(params_rng, dropout_rng, u, init_data))
  return model, variables


def get_dequantizer(model, variables, train=False):
  def dequantizer(u, x, rng=None):
    if not train:
      u_deq, sldj = model.apply(variables, u, x, train=train, inverse=False)
    else:
      u_deq, sldj = model.apply(variables, u, x, train=train, inverse=False, rngs={'dropout': rng})

    return u_deq, sldj

  return dequantizer


def get_model_fn(model, params, states, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all mutable states.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels, rng=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      rng: If present, it is the random state for dropout

    Returns:
      A tuple of (model output, new mutable states)
    """
    variables = {'params': params, **states}
    if not train:
      return model.apply(variables, x, labels, train=False, mutable=False), states
    else:
      rngs = {'dropout': rng}
      return model.apply(variables, x, labels, train=True, mutable=list(states.keys()), rngs=rngs)
      # if states:
      #   return outputs
      # else:
      #   return outputs, states

  return model_fn


def get_score_fn(sde, model, params, states, train=False, continuous=False, return_state=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all other mutable parameters.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    return_state: If `True`, return the new mutable states alongside the model output.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, params, states, train=train)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t, rng=None):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        model, state = model_fn(x, labels, rng)
        std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        model, state = model_fn(x, labels, rng)
        std = sde.sqrt_1m_alphas_cumprod[labels.astype(jnp.int32)]

      score = batch_mul(-model, 1. / std)
      if return_state:
        return score, state
      else:
        return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, rng=None):
      if sde.linear is False:
        if continuous:
          labels = sde.marginal_prob(jnp.zeros_like(x), t)[1]
        else:
          # For VE-trained models, t=0 corresponds to the highest noise level
          labels = sde.T - t
          labels *= sde.N - 1
          labels = jnp.round(labels).astype(jnp.int32)

        score, state = model_fn(x, labels, rng)
      else:
        assert continuous
        labels = t * 999
        model, state = model_fn(x, labels, rng)
        std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
        score = batch_mul(-model, 1. / std)

      if return_state:
        return score, state
      else:
        return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a JAX array `x` and convert it to numpy."""
  return np.asarray(x.reshape((-1,)), dtype=np.float64)


def from_flattened_numpy(x, shape):
  """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
  return jnp.asarray(x, dtype=jnp.float32).reshape(shape)