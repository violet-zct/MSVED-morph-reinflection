__author__ = 'chuntingzhou'
import numpy as np
import theano
import theano.tensor as T
import math


def get_kl_weight(update_ind, thres, rate):
    upnum = 10000
    if update_ind <= upnum:
        return 0.0
    else:
        w = (1.0/rate)*(update_ind - upnum)
        if w < thres:
            return w
        else:
            return thres


def get_sl_weight(update_ind, thres=1.0, rate=10000.0):
    upnum = 0
    if update_ind <= upnum:
        return 0.0
    else:
        w = (1.0/rate)*(update_ind - upnum)
        if w < thres:
            return w
        else:
            return thres

def get_temp(update_ind):
    return max(0.5, math.exp(-3 * 1e-5 * update_ind))


def sample_gumbel(shape, rng, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = rng.uniform(shape, low=0.0, high=1.0)
    return -T.log(-T.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, rng):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(logits.shape, rng)
  return T.nnet.softmax(y / temperature)


def gumbel_softmax(logits, temperature, rng, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  # T.equal(y, T.argmax(y, axis=1, keep_dims=True)
  y = gumbel_softmax_sample(logits, temperature, rng)
  if hard:
    y_hard = T.cast(T.eq(y, T.max(y, axis=1, keepdims=True)), theano.config.floatX)
    y = theano.gradient.disconnected_grad(y_hard - y) + y
  return y


def standard_normal(x):
    c = - 0.5 * T.log(2 * T.constant(np.pi))
    return c - x ** 2 / 2


def normal2(x, mean, logvar):
    c = - 0.5 * T.log(2 * T.constant(np.pi))
    return c - logvar / 2 - (x - mean) ** 2 / (2 * T.exp(logvar))
