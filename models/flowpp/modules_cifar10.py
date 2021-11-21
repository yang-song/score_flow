"""
Ported from https://github.com/aravindsrinivas/flowpp/blob/737fadb2218c1e2810a91b523498f97def2c30de/flows/flows.py
"""

import flax
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Any
from .logistic import *
from jax.experimental import host_callback


def safe_log(x):
  return jnp.log(jnp.clip(x, a_min=1e-7))


class CheckerboardSplit(nn.Module):
  inverse_module: bool = False

  @nn.compact
  def __call__(self, x, inverse=False):
    if self.inverse_module:
      inverse = not inverse

    if not inverse:
      B, H, W, C = x.shape
      x = jnp.reshape(x, [B, H, W // 2, 2, C])
      a = x[:, :, :, 0, :]
      b = x[:, :, :, 1, :]
      assert a.shape == b.shape == (B, H, W // 2, C)
      return (a, b), None
    else:
      a, b = x
      assert a.shape == b.shape
      B, H, W_half, C = a.shape
      y = jnp.stack([a, b], axis=3)
      assert y.shape == (B, H, W_half, 2, C)
      return jnp.reshape(y, [B, H, W_half * 2, C]), None


def init_normalization(self, x, *, name, init_scale=1.):
  def g_initializer(rng, x):
    # data based normalization
    # v_init = jnp.var(x, axis=0)
    m_init = jax.lax.pmean(jnp.mean(x, axis=0), axis_name='batch')
    m2_init = jax.lax.pmean(jnp.mean(x ** 2, axis=0), axis_name='batch')
    v_init = m2_init - m_init ** 2
    return init_scale * jax.lax.rsqrt(v_init + 1e-6)

  def b_initializer(rng, x):
    # data based normalization
    # m_init = jnp.mean(x, axis=0)
    # v_init = jnp.var(x, axis=0)

    m_init = jax.lax.pmean(jnp.mean(x, axis=0), axis_name='batch')
    m2_init = jax.lax.pmean(jnp.mean(x ** 2, axis=0), axis_name='batch')
    v_init = m2_init - m_init ** 2

    scale_init = init_scale * jax.lax.rsqrt(v_init + 1e-6)
    assert m_init.shape == v_init.shape == scale_init.shape

    return -m_init * scale_init

  g = self.param(f'{name}_g', g_initializer, x)
  b = self.param(f'{name}_b', b_initializer, x)

  return g, b


class Norm(nn.Module):
  init_scale: float = 1.

  @nn.compact
  def __call__(self, inputs, inverse=False):
    assert not isinstance(inputs, list)
    if isinstance(inputs, tuple):
      is_tuple = True
    else:
      inputs = [inputs]
      is_tuple = False

    bs = int(inputs[0].shape[0])
    g_and_b = []
    for (i, x) in enumerate(inputs):
      g, b = init_normalization(self, x, name='norm{}'.format(i), init_scale=self.init_scale)
      g = jnp.maximum(g, 1e-10)
      assert x.shape[0] == bs and g.shape == b.shape == x.shape[1:]
      g_and_b.append((g, b))

    logd = jnp.full([bs], sum([jnp.sum(safe_log(g)) for (g, _) in g_and_b]))
    if not inverse:
      out = [x * g[None] + b[None] for (x, (g, b)) in zip(inputs, g_and_b)]
    else:
      out = [(x - b[None]) / g[None] for (x, (g, b)) in zip(inputs, g_and_b)]
      logd = -logd

    if not is_tuple:
      assert len(out) == 1
      return out[0], logd
    return tuple(out), logd


class Pointwise(nn.Module):
  noisy_identity_init: float = 0.001

  @nn.compact
  def __call__(self, inputs, noisy_identity_init=0.001, inverse=False):
    assert not isinstance(inputs, list)
    if isinstance(inputs, tuple):
      is_tuple = True
    else:
      inputs = [inputs]
      is_tuple = False

    out, logds = [], []
    for i, x in enumerate(inputs):
      if self.noisy_identity_init:
        # identity + gaussian noise
        def initializer(key, x):
          _, img_h, img_w, img_c = x.shape
          return jnp.eye(img_c) + self.noisy_identity_init * jax.random.normal(key, (img_c, img_c))
      else:
        # random orthogonal
        def initializer(key, x):
          _, img_h, img_w, img_c = x.shape
          return jnp.linalg.qr(jax.random.normal(key, (img_c, img_c)))[0]

      W = self.param('W{}'.format(i), initializer, x)
      out.append(self._nin(x, W if not inverse else jnp.linalg.inv(W)))
      _, img_h, img_w, img_c = x.shape
      logds.append((1 if not inverse else -1) * img_h * img_w * jnp.linalg.slogdet(W)[1])
    logd = jnp.full([inputs[0].shape[0]], sum(logds))

    if not is_tuple:
      assert len(out) == 1
      return out[0], logd
    return tuple(out), logd

  @staticmethod
  def _nin(x, w, b=None):
    _, out_dim = w.shape
    s = x.shape
    x = jnp.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = x @ w
    if b is not None:
      assert len(b.shape) == 1
      x = x + b[None, :]
    return jnp.reshape(x, s[:-1] + (out_dim,))


def conv2d(self, x, *, name, num_units, filter_size=(3, 3), stride=(1, 1), pad='SAME', init_scale=1.):
  assert len(x.shape) == 4

  def W_initializer(rng, x):
    W = jax.random.normal(rng, [*filter_size, int(x.shape[-1]), num_units]) * 0.05
    y = jax.lax.conv_general_dilated(x, W, window_strides=stride, padding=pad,
                                     dimension_numbers=('NHWC', 'HWIO', 'NWHC'))
    # v_init = jnp.var(y, axis=(0, 1, 2))
    m_init = jax.lax.pmean(jnp.mean(y, axis=(0, 1, 2)), axis_name='batch')
    m2_init = jax.lax.pmean(jnp.mean(y ** 2, axis=(0, 1, 2)), axis_name='batch')
    v_init = m2_init - m_init ** 2
    scale_init = init_scale * jax.lax.rsqrt(v_init + 1e-6)

    return W * scale_init[None, None, None, :]

  def b_initializer(rng, x):
    W = jax.random.normal(rng, [*filter_size, int(x.shape[-1]), num_units]) * 0.05
    y = jax.lax.conv_general_dilated(x, W, window_strides=stride, padding=pad,
                                     dimension_numbers=('NHWC', 'HWIO', 'NWHC'))
    # m_init = jnp.mean(y, axis=(0, 1, 2))
    # v_init = jnp.var(y, axis=(0, 1, 2))
    m_init = jax.lax.pmean(jnp.mean(y, axis=(0, 1, 2)), axis_name='batch')
    m2_init = jax.lax.pmean(jnp.mean(y ** 2, axis=(0, 1, 2)), axis_name='batch')
    v_init = m2_init - m_init ** 2
    scale_init = init_scale * jax.lax.rsqrt(v_init + 1e-6)

    return -m_init * scale_init

  W = self.param(f'{name}_W', W_initializer, x)
  b = self.param(f'{name}_b', b_initializer, x)

  return jax.lax.conv_general_dilated(x, W, window_strides=stride, padding=pad,
                                      dimension_numbers=('NWHC', 'HWIO', 'NWHC')) + b[None, None, None, :]


def concat_elu(x):
  axis = len(x.shape) - 1
  return jax.nn.elu(jnp.concatenate([x, -x], axis))


def dense(self, x, *, name, num_units, init_scale=1.):
  _, in_dim = x.shape

  def W_initializer(rng, x):
    W = jax.random.normal(rng, [in_dim, num_units]) * 0.05
    y = x @ W
    # v_init = jnp.var(y, axis=0)
    m_init = jax.lax.pmean(jnp.mean(y, axis=0), axis_name='batch')
    m2_init = jax.lax.pmean(jnp.mean(y ** 2, axis=0), axis_name='batch')
    v_init = m2_init - m_init ** 2
    scale_init = init_scale * jax.lax.rsqrt(v_init + 1e-6)
    return W * scale_init[None, :]

  def b_initializer(rng, x):
    W = jax.random.normal(rng, [in_dim, num_units]) * 0.05
    y = x @ W
    # m_init = jnp.mean(y, axis=0)
    # v_init = jnp.var(y, axis=0)
    m_init = jax.lax.pmean(jnp.mean(y, axis=0), axis_name='batch')
    m2_init = jax.lax.pmean(jnp.mean(y ** 2, axis=0), axis_name='batch')
    v_init = m2_init - m_init ** 2
    scale_init = init_scale * jax.lax.rsqrt(v_init + 1e-6)
    return -m_init * scale_init

  W = self.param(f'{name}_W', W_initializer, x)
  b = self.param(f'{name}_b', b_initializer, x)

  return x @ W + b[None, :]


def nin(self, x, *, num_units, **kwargs):
  assert 'num_units' not in kwargs
  s = x.shape
  x = jnp.reshape(x, [np.prod(s[:-1]), s[-1]])
  x = dense(self, x, num_units=num_units, **kwargs)
  return jnp.reshape(x, s[:-1] + (num_units,))


def gate(x, *, axis):
  a, b = jnp.split(x, 2, axis=axis)
  return a * jax.nn.sigmoid(b)


def gated_conv(self, x, *, name, a, nonlinearity=concat_elu, conv=conv2d, use_nin, dropout_p, train=False):
  num_filters = int(x.shape[-1])

  c1 = conv(self, nonlinearity(x), name=f'{name}_c1', num_units=num_filters)
  if a is not None:  # add short-cut connection if auxiliary input 'a' is given
    c1 += nin(self, nonlinearity(a), name=f'{name}_a_proj', num_units=num_filters)
  c1 = nonlinearity(c1)
  if dropout_p > 0:
    c1 = nn.Dropout(rate=dropout_p, deterministic=not train)(c1)

  c2 = (nin if use_nin else conv)(self, c1, name=f'{name}_c2', num_units=num_filters * 2, init_scale=0.1)
  return x + gate(c2, axis=3)


def layernorm(self, x, *, name, e=1e-5):
  """Layer norm over last axis"""
  shape = [1] * (len(x.shape) - 1) + [int(x.shape[-1])]
  g = self.param(f'{name}_g', jax.nn.initializers.ones, shape)
  b = self.param(f'{name}_b', jax.nn.initializers.zeros, shape)
  u = jnp.mean(x, axis=-1, keepdims=True)
  s = jnp.mean(jnp.square(x - u), axis=-1, keepdims=True)
  return (x - u) * jax.lax.rsqrt(s + e) * g + b


def gated_attn(self, x, *, name, pos_emb, heads, dropout_p, train=False):
  bs, height, width, ch = x.shape
  assert pos_emb.shape == (height, width, ch)
  assert ch % heads == 0
  timesteps = height * width
  dim = ch // heads
  # Position embeddings
  c = x + pos_emb[None, :, :, :]
  # b, h, t, d == batch, num heads, num timesteps, per-head dim (C // heads)
  c = nin(self, c, name=f'{name}_proj1', num_units=3 * ch)
  assert c.shape == (bs, height, width, 3 * ch)
  # Split into heads / Q / K / V
  c = jnp.reshape(c, [bs, timesteps, 3, heads, dim])  # b, t, 3, h, d
  c = jnp.transpose(c, [2, 0, 3, 1, 4])  # 3, b, h, t, d
  q_bhtd, k_bhtd, v_bhtd = c[0, ...], c[1, ...], c[2, ...]
  assert q_bhtd.shape == k_bhtd.shape == v_bhtd.shape == (bs, heads, timesteps, dim)
  # Attention
  w_bhtt = jnp.einsum('bhTD,bhtD->bhTt', q_bhtd, k_bhtd) / np.sqrt(float(dim))
  w_bhtt = jax.nn.softmax(w_bhtt)
  assert w_bhtt.shape == (bs, heads, timesteps, timesteps)
  a_bhtd = jnp.einsum('bhTt,bhtd->bhTd', w_bhtt, v_bhtd)
  # Merge heads
  a_bthd = jnp.transpose(a_bhtd, [0, 2, 1, 3])
  assert a_bthd.shape == (bs, timesteps, heads, dim)
  a_btc = jnp.reshape(a_bthd, [bs, timesteps, ch])
  # Project
  c1 = jnp.reshape(a_btc, [bs, height, width, ch])
  if dropout_p > 0:
    c1 = nn.Dropout(rate=dropout_p, deterministic=not train)(c1)
  c2 = nin(self, c1, name=f'{name}_proj2', num_units=ch * 2, init_scale=0.1)
  return x + gate(c2, axis=3)


def sumflat(x):
  return jnp.sum(jnp.reshape(x, [x.shape[0], -1]), axis=1)


def inverse_sigmoid(x):
  return -safe_log(jax.lax.reciprocal(x) - 1.)


class Sigmoid(nn.Module):
  inverse_module: bool = False

  @nn.compact
  def __call__(self, x, inverse=False):
    if self.inverse_module:
      inverse = not inverse
    if not inverse:
      y = jax.nn.sigmoid(x)
      logd = -jax.nn.softplus(x) - jax.nn.softplus(-x)
      return y, sumflat(logd)
    else:
      y = inverse_sigmoid(x)
      logd = -safe_log(x) - safe_log(1. - x)
      return y, sumflat(logd)


class MixLogisticCDF(nn.Module):
  """
  Elementwise transformation by the CDF of a mixture of logistics
  """
  min_logscale: float = -7.

  @nn.compact
  def __call__(self, x, logits, means, logscales, inverse=False):
    logistic_kwargs = dict(
      prior_logits=logits,
      means=means,
      logscales=jnp.maximum(logscales, self.min_logscale)
    )
    if not inverse:
      out = jnp.exp(mixlogistic_logcdf(x=x, **logistic_kwargs))
      logd = mixlogistic_logpdf(x=x, **logistic_kwargs)
      return out, sumflat(logd)
    else:
      out = mixlogistic_invcdf(y=jnp.clip(x, 0., 1.), **logistic_kwargs)
      logd = -mixlogistic_logpdf(x=out, **logistic_kwargs)
      return out, sumflat(logd)


class ElemwiseAffine(nn.Module):

  @nn.compact
  def __call__(self, x, scales, biases, logscales=None, inverse=False):
    logscales = safe_log(scales) if logscales is None else logscales
    if not inverse:
      assert logscales.shape == x.shape
      return (x * scales + biases), sumflat(logscales)
    else:
      y = x
      assert logscales.shape == y.shape
      return ((y - biases) / scales), sumflat(-logscales)


class MixLogisticAttnCoupling(nn.Module):
  """
  CDF of mixture of logistics, followed by affine
  """
  filters: int
  blocks: int
  components: int
  heads: int = 4
  init_scale: float = 0.1
  dropout_p: float = 0.
  verbose: bool = True

  @nn.compact
  def __call__(self, x, context=None, inverse=False, train=False):
    def f(x, *, context=None):
      if not self.has_variable('params', 'pos_emb') and self.verbose:
        # debug stuff
        def tap_func(x, transforms):
          xmean = jnp.mean(x, axis=list(range(len(x.shape))))
          xvar = jnp.var(x, axis=list(range(len(x.shape))))
          print(f'shape: {jnp.shape(x)}')
          print(f'mean: {xmean}')
          print(f'std: {jnp.sqrt(xvar)}')
          print(f'min: {jnp.min(x)}')
          print(f'max: {jnp.max(x)}')

        x = host_callback.id_tap(tap_func, x)

      B, H, W, C = x.shape
      pos_emb = self.param('pos_emb', jax.nn.initializers.normal(stddev=0.01), [H, W, self.filters])
      x = conv2d(self, x, name='proj_in', num_units=self.filters)
      for i_block in range(self.blocks):
        name = f'block{i_block}'
        x = gated_conv(self, x, name=f'{name}_conv', a=context, use_nin=True, dropout_p=self.dropout_p, train=train)
        x = layernorm(self, x, name=f'{name}_ln1')
        x = gated_attn(self, x, name=f'{name}_attn', pos_emb=pos_emb, heads=self.heads, dropout_p=self.dropout_p,
                       train=train)
        x = layernorm(self, x, name=f'{name}_ln2')
      x = conv2d(self, x, name=f'{name}_proj_out', num_units=C * (2 + 3 * self.components), init_scale=self.init_scale)
      assert x.shape == (B, H, W, C * (2 + 3 * self.components))
      x = jnp.reshape(x, [B, H, W, C, 2 + 3 * self.components])

      s, t = jnp.tanh(x[:, :, :, :, 0]), x[:, :, :, :, 1]
      ml_logits, ml_means, ml_logscales = jnp.split(x[:, :, :, :, 2:], 3, axis=4)
      assert s.shape == t.shape == (B, H, W, C)
      assert ml_logits.shape == ml_means.shape == ml_logscales.shape == (B, H, W, C, self.components)
      return ml_logits, ml_means, ml_logscales, s, t

    assert isinstance(x, tuple)
    cf, ef = x
    ml_logits, ml_means, ml_logscales, s, t = f(cf, context=context)
    logp_sum = 0.

    mixlogistic_cdf = MixLogisticCDF()
    sigmoid = Sigmoid(inverse_module=True)
    elementwise_affine = ElemwiseAffine()

    if not inverse:
      h, logp = mixlogistic_cdf(ef, logits=ml_logits, means=ml_means, logscales=ml_logscales, inverse=False)
      if logp is not None:
        logp_sum = logp_sum + logp
      h, logp = sigmoid(h, inverse=False)
      if logp is not None:
        logp_sum = logp_sum + logp
      h, logp = elementwise_affine(h, scales=jnp.exp(s), biases=t, logscales=s, inverse=False)
      if logp is not None:
        logp_sum = logp_sum + logp
      return (cf, h), logp_sum

    else:
      h, logp = elementwise_affine(ef, scales=jnp.exp(s), biases=t, logscales=s, inverse=True)
      if logp is not None:
        logp_sum = logp_sum + logp
      h, logp = sigmoid(h, inverse=True)
      if logp is not None:
        logp_sum = logp_sum + logp
      h, logp = mixlogistic_cdf(h, logits=ml_logits, means=ml_means, logscales=ml_logscales, inverse=True)
      if logp is not None:
        logp_sum = logp_sum + logp
      return (cf, h), logp_sum


class TupleFlip(nn.Module):
  @nn.compact
  def __call__(self, x, inverse=False):
    assert isinstance(x, tuple)
    a, b = x
    return (b, a), None
