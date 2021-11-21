import jax
import jax.numpy as jnp
import flax.linen as nn
from .modules_cifar10 import concat_elu, nin, gate, layernorm as norm, MixLogisticCDF, Sigmoid, ElemwiseAffine, conv2d
from jax.experimental import host_callback
from typing import Any


def gated_resnet(self, x, *, name, a, nonlinearity=concat_elu, conv=conv2d, use_nin, dropout_p, train=False):
  num_filters = int(x.shape[-1])

  c1 = conv(self, nonlinearity(x), name=f'{name}_c1', num_units=num_filters)
  if a is not None:  # add short-cut connection if auxiliary input 'a' is given
    c1 += nin(self, nonlinearity(a), name=f'{name}_a_proj', num_units=num_filters)
  c1 = nonlinearity(c1)
  if dropout_p > 0:
    c1 = nn.Dropout(rate=dropout_p, deterministic=not train)(c1)

  c2 = (nin if use_nin else conv)(self, c1, name='c2', num_units=num_filters * 2, init_scale=0.1)
  return x + gate(c2, axis=3)


class MixLogisticCoupling(nn.Module):
  """
  CDF of mixture of logistics, followed by affine
  """
  filters: int
  blocks: int
  components: int
  heads: int = 4
  init_scale: float = 0.1
  dropout_p: float = 0.
  use_nin: bool = True
  use_ln: bool = True
  with_affine: bool = True
  use_final_nin: bool = False
  nonlinearity: Any = concat_elu
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
      x = conv2d(self, x, name='c1', num_units=self.filters)
      for i_block in range(self.blocks):
        name = f'block{i_block}'
        x = gated_resnet(self, x, name=f'{name}_conv', a=context, use_nin=self.use_nin, dropout_p=self.dropout_p, train=train)
        if self.use_ln:
          x = norm(self, x, name=f'{name}_ln1')

      x = self.nonlinearity(x)
      x = (nin if self.use_final_nin else conv2d)(
        self, x, name=f'{name}_c2', num_units=C * (2 + 3 * self.components), init_scale=self.init_scale)

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
    if self.with_affine:
      elementwise_affine = ElemwiseAffine()

    if not inverse:
      h, logp = mixlogistic_cdf(ef, logits=ml_logits, means=ml_means, logscales=ml_logscales, inverse=False)
      if logp is not None:
        logp_sum = logp_sum + logp
      h, logp = sigmoid(h, inverse=False)
      if logp is not None:
        logp_sum = logp_sum + logp
      if self.with_affine:
        h, logp = elementwise_affine(h, scales=jnp.exp(s), biases=t, logscales=s, inverse=False)
        if logp is not None:
          logp_sum = logp_sum + logp
      return (cf, h), logp_sum

    else:
      if self.with_affine:
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