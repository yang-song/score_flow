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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import jax
import jax.numpy as jnp
import numpy as np

import utils
from utils import batch_mul
from models import utils as mutils
from utils import get_div_fn, get_value_div_fn


def get_likelihood_bound_fn(sde, model, inverse_scaler, hutchinson_type='Rademacher',
                            dsm=True, eps=1e-5, N=1000, importance_weighting=True,
                            eps_offset=True):
  """Create a function to compute the unbiased log-likelihood bound of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    dsm: bool. Use denoising score matching bound if enabled; otherwise use sliced score matching.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.
    N: The number of time values to be sampled.
    importance_weighting: True if enable importance weighting for potential variance reduction.
    eps_offset: True if use Jensen's inequality to offset the likelihood bound due to non-zero starting time.

  Returns:
    A function that takes random states, replicated training states, and a batch of data points
      and returns the log-likelihoods in bits/dim, the latent code, and the number of function
      evaluations cost by computation.
  """

  def value_div_score_fn(state, x, t, eps):
    """Pmapped divergence of the drift function."""
    score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    value_div_fn = get_value_div_fn(lambda x, t: score_fn(x, t))
    return value_div_fn(x, t, eps)

  def div_drift_fn(x, t, eps):
    div_fn = get_div_fn(lambda x, t: sde.sde(x, t)[0])
    return div_fn(x, t, eps)

  def likelihood_bound_fn(prng, state, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      prng: An array of random states. The list dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      data: A JAX array of shape [#devices, batch size, ...].

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      N: same as input
    """
    rng, step_rng = jax.random.split(prng)
    if importance_weighting:
      time_samples = sde.sample_importance_weighted_time_for_likelihood(step_rng, (N, data.shape[0]), eps=eps)
      Z = sde.likelihood_importance_cum_weight(sde.T, eps=eps)
    else:
      time_samples = jax.random.uniform(step_rng, (N, data.shape[0]), minval=eps, maxval=sde.T)
      Z = 1

    shape = data.shape
    if not dsm:
      def scan_fn(carry, vec_time):
        rng, value = carry
        rng, step_rng = jax.random.split(rng)
        if hutchinson_type == 'Gaussian':
          epsilon = jax.random.normal(step_rng, shape)
        elif hutchinson_type == 'Rademacher':
          epsilon = jax.random.rademacher(step_rng, shape, dtype=jnp.float32)
        else:
          raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

        rng, step_rng = jax.random.split(rng)
        noise = jax.random.normal(step_rng, shape)
        mean, std = sde.marginal_prob(data, vec_time)
        noisy_data = mean + utils.batch_mul(std, noise)
        score_val, score_div = value_div_score_fn(state, noisy_data, vec_time, epsilon)
        score_norm = jnp.square(score_val.reshape((score_val.shape[0], -1))).sum(axis=-1)
        drift_div = div_drift_fn(noisy_data, vec_time, epsilon)
        f, g = sde.sde(noisy_data, vec_time)
        integrand = utils.batch_mul(g ** 2, 2 * score_div + score_norm) - 2 * drift_div
        if importance_weighting:
          integrand = utils.batch_mul(std ** 2 / g ** 2 * Z, integrand)
        return (rng, value + integrand), integrand
    else:
      score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)

      def scan_fn(carry, vec_time):
        rng, value = carry
        rng, step_rng = jax.random.split(rng)
        if hutchinson_type == 'Gaussian':
          epsilon = jax.random.normal(step_rng, shape)
        elif hutchinson_type == 'Rademacher':
          epsilon = jax.random.rademacher(step_rng, shape, dtype=jnp.float32)
        else:
          raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
        rng, step_rng = jax.random.split(rng)
        noise = jax.random.normal(step_rng, shape)
        mean, std = sde.marginal_prob(data, vec_time)
        noisy_data = mean + utils.batch_mul(std, noise)
        drift_div = div_drift_fn(noisy_data, vec_time, epsilon)
        score_val = score_fn(noisy_data, vec_time)
        grad = utils.batch_mul(-(noisy_data - mean), 1 / std ** 2)
        diff1 = score_val - grad
        diff1 = jnp.square(diff1.reshape((diff1.shape[0], -1))).sum(axis=-1)
        diff2 = jnp.square(grad.reshape((grad.shape[0], -1))).sum(axis=-1)
        f, g = sde.sde(noisy_data, vec_time)
        integrand = utils.batch_mul(g ** 2, diff1 - diff2) - 2 * drift_div
        if importance_weighting:
          integrand = utils.batch_mul(std ** 2 / g ** 2 * Z, integrand)
        return (rng, value + integrand), integrand

    (rng, integral), _ = jax.lax.scan(scan_fn, (rng, jnp.zeros((shape[0],))), time_samples)
    integral = integral / N
    mean, std = sde.marginal_prob(data, jnp.ones((data.shape[0],)) * sde.T)
    rng, step_rng = jax.random.split(rng)
    noise = jax.random.normal(step_rng, shape)
    neg_prior_logp = -sde.prior_logp(mean + utils.batch_mul(std, noise))
    nlogp = neg_prior_logp + 0.5 * integral

    # whether to enable likelihood offset
    if eps_offset:
      score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
      offset_fn = get_likelihood_offset_fn(sde, score_fn, eps)
      rng, step_rng = jax.random.split(rng)
      nlogp = nlogp + offset_fn(step_rng, data)

    bpd = nlogp / np.log(2)
    dim = np.prod(shape[1:])
    bpd = bpd / dim

    # A hack to convert log-likelihoods to bits/dim
    # based on the gradient of the inverse data normalizer.
    offset = jnp.log2(jax.grad(inverse_scaler)(0.)) + 8.
    bpd += offset

    return bpd, N

  return jax.pmap(likelihood_bound_fn, axis_name='batch')


def get_likelihood_offset_fn(sde, score_fn, eps=1e-5):
  """Create a function to compute the unbiased log-likelihood bound of a given data point.
  """

  def likelihood_offset_fn(prng, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      prng: An array of random states. The list dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      data: A JAX array of shape [#devices, batch size, ...].

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      N: same as input
    """
    rng, step_rng = jax.random.split(prng)
    shape = data.shape

    eps_vec = jnp.full((shape[0],), eps)
    p_mean, p_std = sde.marginal_prob(data, eps_vec)
    rng, step_rng = jax.random.split(rng)
    noisy_data = p_mean + batch_mul(p_std, jax.random.normal(step_rng, shape))
    score = score_fn(noisy_data, eps_vec)

    alpha, beta = sde.marginal_prob(jnp.ones_like(data), eps_vec)
    q_mean = noisy_data / alpha + batch_mul(beta ** 2, score / alpha)
    q_std = beta / jnp.mean(alpha, axis=(1, 2, 3))

    n_dim = np.prod(data.shape[1:])
    p_entropy = n_dim / 2. * (np.log(2 * np.pi) + 2 * jnp.log(p_std) + 1.)
    q_recon = n_dim / 2. * (np.log(2 * np.pi) + 2 * jnp.log(q_std)) + batch_mul(0.5 / (q_std ** 2),
                                                                                jnp.square(data - q_mean).sum(
                                                                                  axis=(1, 2, 3)))
    offset = q_recon - p_entropy
    return offset

  return likelihood_offset_fn
