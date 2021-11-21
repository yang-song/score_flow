from .modules_imagenet32 import *
from .modules_cifar10 import CheckerboardSplit, Norm, TupleFlip
import numpy as np


class DeepProcessor(nn.Module):
  dropout_p: float = 0.

  @nn.compact
  def __call__(self, x, train=False):
    # x is assumed to take values in [0, 1]
    x = x - 0.5
    (this, that), _ = CheckerboardSplit()(x, inverse=False)
    processed_context = conv2d(self, jnp.concatenate([this, that], axis=3), name='proj', num_units=32)
    B, H, W, C = processed_context.shape
    pos_emb = self.param('pos_emb_dq', jax.nn.initializers.normal(0.01), (H, W, C))

    for i in range(8):
      processed_context = gated_resnet(self, processed_context, name=f'c{i}',
                                       dropout_p=self.dropout_p, use_nin=False, a=None, train=train)
      processed_context = norm(self, processed_context, name=f'dqln{i}')
      processed_context = attn(self, processed_context, name=f'dqattn{i}', pos_emb=pos_emb, heads=4,
                               dropout_p=self.dropout_p, train=train)
      processed_context = norm(self, processed_context, name=f'ln{i}')

    return processed_context


class Dequantization(nn.Module):
  filters: int = 128
  components: int = 32
  blocks: int = 8
  attn_heads: int = 4
  dropout_p: float = 0.
  use_nin: bool = True
  use_ln: bool = True

  @nn.compact
  def __call__(self, eps, x, inverse=False, train=False):
    # x is assumed to take values in [0, 1]
    logp_eps = jnp.sum(-eps ** 2 / 2. - 0.5 * np.log(2 * np.pi), axis=(1, 2, 3))

    coupling_params = dict(
      filters=self.filters,
      blocks=self.blocks,
      components=self.components,
      heads=self.attn_heads,
      use_nin=self.use_nin,
      use_ln=self.use_ln
    )
    modules = [
      CheckerboardSplit(),
      Norm(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      Norm(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      Norm(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      Norm(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      CheckerboardSplit(inverse_module=True),
      CheckerboardSplit(),
      Norm(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      Norm(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      Norm(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      Norm(), MixLogisticAttnCoupling(**coupling_params), TupleFlip(),
      CheckerboardSplit(inverse_module=True),
      Sigmoid()
    ]

    context = DeepProcessor(dropout_p=self.dropout_p)(x, train=train)

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
