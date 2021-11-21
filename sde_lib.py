"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import jax.numpy as jnp
import jax
import numpy as np
from utils import batch_mul


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, rng, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a JAX tensor.
      t: a JAX float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * jnp.sqrt(dt)
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - batch_mul(diffusion ** 2, score * (0.5 if self.probability_flow else 1.))
        # Set the diffusion function to zero for ODEs.
        diffusion = jnp.zeros_like(t) if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - batch_mul(G ** 2, score_fn(x, t) * (0.5 if self.probability_flow else 1.))
        rev_G = jnp.zeros_like(t) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = jnp.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * batch_mul(beta_t, x)
    diffusion = jnp.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t, high_precision=True):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    if high_precision:
      mean = batch_mul(jnp.where(jnp.abs(log_mean_coeff) <= 1e-3, 1 + log_mean_coeff, jnp.exp(log_mean_coeff)), x)
      std = jnp.where(jnp.abs(log_mean_coeff) <= 1e-3, jnp.sqrt(-2. * log_mean_coeff),
                      jnp.sqrt(1 - jnp.exp(2. * log_mean_coeff)))
    else:
      mean = batch_mul(jnp.exp(log_mean_coeff), x)
      std = jnp.sqrt(1 - jnp.exp(2 * log_mean_coeff))
    return mean, std

  def prior_sampling(self, rng, shape):
    return jax.random.normal(rng, shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logp_fn = lambda z: -N / 2. * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.
    return jax.vmap(logp_fn)(z)

  def prior_entropy(self, z):
    shape = z.shape
    entropy = jnp.ones(shape) * (0.5 * jnp.log(2 * np.pi) + 0.5)
    entropy = entropy.reshape((z.shape[0], -1))
    return jnp.sum(entropy, axis=-1)

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
    beta = self.discrete_betas[timestep]
    alpha = self.alphas[timestep]
    sqrt_beta = jnp.sqrt(beta)
    f = batch_mul(jnp.sqrt(alpha), x) - x
    G = sqrt_beta
    return f, G

  def likelihood_importance_cum_weight(self, t, eps=1e-5):
    exponent1 = 0.5 * eps * (eps - 2) * self.beta_0 - 0.5 * eps ** 2 * self.beta_1
    exponent2 = 0.5 * t * (t - 2) * self.beta_0 - 0.5 * t ** 2 * self.beta_1
    term1 = jnp.where(jnp.abs(exponent1) <= 1e-3, -exponent1, 1. - jnp.exp(exponent1))
    term2 = jnp.where(jnp.abs(exponent2) <= 1e-3, -exponent2, 1. - jnp.exp(exponent2))
    return 0.5 * (-2 * jnp.log(term1) + 2 * jnp.log(term2)
                  + self.beta_0 * (-2 * eps + eps ** 2 - (t - 2) * t)
                  + self.beta_1 * (-eps ** 2 + t ** 2))

  def sample_importance_weighted_time_for_likelihood(self, rng, shape, quantile=None, eps=1e-5, steps=100):
    Z = self.likelihood_importance_cum_weight(self.T, eps=eps)
    if quantile is None:
      quantile = jax.random.uniform(rng, shape, minval=0, maxval=Z)
    lb = jnp.ones_like(quantile) * eps
    ub = jnp.ones_like(quantile) * self.T

    def bisection_func(carry, idx):
      lb, ub = carry
      mid = (lb + ub) / 2.
      value = self.likelihood_importance_cum_weight(mid, eps=eps)
      lb = jnp.where(value <= quantile, mid, lb)
      ub = jnp.where(value <= quantile, ub, mid)
      return (lb, ub), idx

    (lb, ub), _ = jax.lax.scan(bisection_func, (lb, ub), jnp.arange(0, steps))
    return (lb + ub) / 2.


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t, high_precision=True):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * batch_mul(beta_t, x)
    exponent = -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2
    discount = 1. - jnp.exp(exponent)
    if high_precision:
      discount = jnp.where(jnp.abs(exponent) <= 1e-3, -exponent, discount)
    diffusion = jnp.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t, high_precision=True):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    if high_precision:
      mean = batch_mul(jnp.where(jnp.abs(log_mean_coeff) <= 1e-3, 1. + log_mean_coeff, jnp.exp(log_mean_coeff)), x)
      std = jnp.where(jnp.abs(log_mean_coeff) <= 1e-3, -2. * log_mean_coeff, 1 - jnp.exp(2. * log_mean_coeff))
    else:
      mean = batch_mul(jnp.exp(log_mean_coeff), x)
      std = 1 - jnp.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, rng, shape):
    return jax.random.normal(rng, shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logp_fn = lambda z: -N / 2. * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.
    return jax.vmap(logp_fn)(z)

  def prior_entropy(self, z):
    shape = z.shape
    entropy = jnp.ones(shape) * (0.5 * jnp.log(2 * np.pi) + 0.5)
    entropy = entropy.reshape((z.shape[0], -1))
    return jnp.sum(entropy, axis=-1)

  def likelihood_importance_cum_weight(self, t, eps=1e-5):
    exponent1 = 0.5 * eps * (eps * self.beta_1 - (eps - 2) * self.beta_0)
    exponent2 = 0.5 * t * (self.beta_1 * t - (t - 2) * self.beta_0)
    term1 = jnp.where(exponent1 <= 1e-3, jnp.log(exponent1), jnp.log(jnp.exp(exponent1) - 1.))
    term2 = jnp.where(exponent2 <= 1e-3, jnp.log(exponent2), jnp.log(jnp.exp(exponent2) - 1.))
    return 0.5 * (-4 * term1 + 4 * term2
                  + (2 * eps - eps ** 2 + t * (t - 2)) * self.beta_0 + (eps ** 2 - t ** 2) * self.beta_1)

  def sample_importance_weighted_time_for_likelihood(self, rng, shape, quantile=None, eps=1e-5, steps=100):
    Z = self.likelihood_importance_cum_weight(self.T, eps=eps)
    if quantile is None:
      quantile = jax.random.uniform(rng, shape, minval=0, maxval=Z)
    lb = jnp.ones_like(quantile) * eps
    ub = jnp.ones_like(quantile) * self.T

    def bisection_func(carry, idx):
      lb, ub = carry
      mid = (lb + ub) / 2.
      value = self.likelihood_importance_cum_weight(mid, eps=eps)
      lb = jnp.where(value <= quantile, mid, lb)
      ub = jnp.where(value <= quantile, ub, mid)
      return (lb, ub), idx

    (lb, ub), _ = jax.lax.scan(bisection_func, (lb, ub), jnp.arange(0, steps))
    return (lb + ub) / 2.


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, linear=False):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.linear = linear
    if not linear:
      self.discrete_sigmas = jnp.exp(np.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    else:
      self.discrete_sigmas = jnp.linspace(self.sigma_min, self.sigma_max, N)
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    drift = jnp.zeros_like(x)
    if not self.linear:
      sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
      diffusion = sigma * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))
    else:
      diffusion = self.sigma_max * jnp.sqrt(2 * t)

    return drift, diffusion

  def marginal_prob(self, x, t):
    mean = x
    if not self.linear:
      std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    else:
      std = t * self.sigma_max
    return mean, std

  def prior_sampling(self, rng, shape):
    return jax.random.normal(rng, shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logp_fn = lambda z: -N / 2. * jnp.log(2 * np.pi * self.sigma_max ** 2) - jnp.sum(z ** 2) / (2 * self.sigma_max ** 2)
    return jax.vmap(logp_fn)(z)

  def prior_entropy(self, z):
    shape = z.shape
    entropy = jnp.ones(shape) * (0.5 * jnp.log(2 * np.pi * self.sigma_max ** 2) + 0.5)
    entropy = entropy.reshape((z.shape[0], -1))
    return jnp.sum(entropy, axis=-1)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    if not self.linear:
      timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
      sigma = self.discrete_sigmas[timestep]
      adjacent_sigma = jnp.where(timestep == 0, jnp.zeros_like(timestep), self.discrete_sigmas[timestep - 1])
      f = jnp.zeros_like(x)
      G = jnp.sqrt(sigma ** 2 - adjacent_sigma ** 2)
      return f, G
    else:
      return super().discretize(x, t)
