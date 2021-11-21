from .modules_cifar10 import *


class ShallowProcessor(nn.Module):
  dropout_p: float = 0.2

  @nn.compact
  def __call__(self, x, train=False):
    # x is assumed to take values in [0, 1]
    x = x - 0.5
    (this, that), _ = CheckerboardSplit()(x, inverse=False)
    x = conv2d(self, jnp.concatenate([this, that], axis=3), name='proj', num_units=32)
    for i in range(3):
      x = gated_conv(self, x, name=f'c{i}', dropout_p=self.dropout_p, use_nin=False, a=None, train=train)
    return x


class Dequantization(nn.Module):
  filters: int = 96
  components: int = 32
  blocks: int = 2
  attn_heads: int = 4
  dropout_p: float = 0.

  @nn.compact
  def __call__(self, eps, x, inverse=False, train=False):
    # x is assumed to take values in [0, 1]
    logp_eps = jnp.sum(-eps ** 2 / 2. - 0.5 * np.log(2 * np.pi), axis=(1, 2, 3))

    coupling_params = dict(
      filters=self.filters,
      blocks=self.blocks,
      components=self.components,
      heads=self.attn_heads
    )
    modules = [
      CheckerboardSplit(),
      Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      CheckerboardSplit(inverse_module=True),
      Sigmoid()
    ]

    context = ShallowProcessor(dropout_p=self.dropout_p)(x, train=train)

    if not inverse:
      logp_sum = 0.
      h = eps
      for module in modules:
        if isinstance(module, MixLogisticAttnCoupling):
          h, logp = module(h, context=context, inverse=inverse, train=train)
        else:
          h, logp = module(h, inverse=inverse)
        logp_sum = logp_sum + logp if logp is not None else logp_sum
      return h, logp_sum - logp_eps

    else:
      logp_sum = 0.
      h = eps
      for module in modules[::-1]:
        if isinstance(module, MixLogisticAttnCoupling):
          h, logp = module(h, context=context, inverse=inverse, train=train)
        else:
          h, logp = module(h, inverse=inverse)
        logp_sum = logp_sum + logp if logp is not None else logp_sum
      return h, logp_sum - logp_eps
