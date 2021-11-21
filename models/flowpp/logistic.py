"""
Ported from https://github.com/aravindsrinivas/flowpp/blob/737fadb2218c1e2810a91b523498f97def2c30de/flows/logistic.py
"""

import jax.numpy as jnp
import jax


def logistic_logpdf(*, x, mean, logscale):
  """
  log density of logistic distribution
  this operates elementwise
  """
  z = (x - mean) * jnp.exp(-logscale)
  return z - logscale - 2 * jax.nn.softplus(z)


def logistic_logcdf(*, x, mean, logscale):
  """
  log cdf of logistic distribution
  this operates elementwise
  """
  z = (x - mean) * jnp.exp(-logscale)
  return jax.nn.log_sigmoid(z)


def mixlogistic_logpdf(*, x, prior_logits, means, logscales):
  """logpdf of a mixture of logistics"""
  assert len(x.shape) + 1 == len(prior_logits.shape) == len(means.shape) == len(logscales.shape)
  return jax.nn.logsumexp(
    jax.nn.log_softmax(prior_logits, axis=-1) + logistic_logpdf(
      x=jnp.expand_dims(x, -1), mean=means, logscale=logscales),
    axis=-1
  )


def mixlogistic_logcdf(*, x, prior_logits, means, logscales):
  """log cumulative distribution function of a mixture of logistics"""
  assert (len(x.shape) + 1 == len(prior_logits.shape) == len(means.shape) == len(logscales.shape))
  return jax.nn.logsumexp(
    jax.nn.log_softmax(prior_logits, axis=-1) + logistic_logcdf(
      x=jnp.expand_dims(x, -1), mean=means, logscale=logscales),
    axis=-1
  )


def mixlogistic_sample(rng, *, prior_logits, means, logscales):
  # Sample mixture component
  rng, step_rng = jax.random.split(rng)
  sampled_inds = jnp.argmax(
    prior_logits - jnp.log(-jnp.log(jax.random.uniform(step_rng, prior_logits.shape,
                                                       minval=1e-5, maxval=1. - 1e-5))),
    axis=-1
  )
  sampled_onehot = jax.nn.one_hot(sampled_inds, prior_logits.shape[-1])
  # Pull out the sampled mixture component
  means = jnp.sum(means * sampled_onehot, axis=-1)
  logscales = jnp.sum(logscales * sampled_onehot, axis=-1)
  # Sample from the component
  rng, step_rng = jax.random.split(rng)
  u = jax.random.uniform(step_rng, means.shape, minval=1e-5, maxval=1. - 1e-5)
  x = means + jnp.exp(logscales) * (jnp.log(u) - jnp.log(1. - u))
  return x


def mixlogistic_invcdf(*, y, prior_logits, means, logscales, tol=1e-12, max_bisection_iters=200,
                       init_bounds_scale=200.):
  """inverse cumulative distribution function of a mixture of logistics"""
  assert len(y.shape) + 1 == len(prior_logits.shape) == len(means.shape) == len(logscales.shape)

  def body(carry, _):
    x, lb, ub = carry
    cur_y = jnp.exp(mixlogistic_logcdf(x=x, prior_logits=prior_logits, means=means, logscales=logscales))
    new_x = jnp.where(cur_y > y, (x + lb) / 2., (x + ub) / 2.)
    new_lb = jnp.where(cur_y > y, lb, x)
    new_ub = jnp.where(cur_y > y, x, ub)
    diff = jnp.max(jnp.abs(new_x - x))
    return (new_x, new_lb, new_ub), diff

  init_x = jnp.zeros_like(y)
  maxscales = jnp.sum(jnp.exp(logscales), axis=-1, keepdims=True)  # sum of scales across mixture components
  init_lb = jnp.min(means - init_bounds_scale * maxscales, axis=-1)
  init_ub = jnp.max(means + init_bounds_scale * maxscales, axis=-1)

  (out_x, _, _), _ = jax.lax.scan(body, (init_x, init_lb, init_ub), jnp.arange(max_bisection_iters),
                                     length=max_bisection_iters)
  assert out_x.shape == y.shape
  return out_x
