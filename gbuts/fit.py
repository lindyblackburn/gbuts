"""data fitting"""

import numpy as np

__version__ = "$Id: fit.py 108 2013-12-16 21:45:21Z lindy.blackburn@LIGO.ORG $"
__author__ = "Lindy Blackburn"
__email__ = "lindy.blackburn@ligo.org"

# binned negative log likelihood based on poisson statistics
def negloglikelihood(par, fun, x, y, w=1):
    import scipy.special
    mu = w * fun(par, x) # allow for weights to be applied to function
    # p = y * np.log(mu) - mu - scipy.special.gammaln(y+1)
    # return -np.sum(np.sort(p)[1:]) # remove 1% outliers
    return -np.sum(y * np.log(mu) - mu - scipy.special.gammaln(y+1))

# double broken powerlaw
# if normed, will integrate to 1 if possible (else crash)
def doublebrokenpowerlaw(par, logx, normed=False):
    (a1, a2, a3, x1, x2) = par # positive power law indices, x1 and x2 are given in log space
    p1 = -a1 * x1
    p2 = -a2 * x1
    p3 = (x2-x1) * a2 - a3 * x2
    line = (logx < x1) * (a1 * logx + p1) + ((x1 <= logx) & (logx < x2)) * (a2 * logx + p2) + (x2 <= logx) * (a3 * logx + p3)
    if normed:
        if(a1 <= -1 or a3 >= -1):
            raise(Exception('bad parameters for normalization'))
        s1 = np.exp(p1) * (np.exp(x1)**(a1+1) - 0**(a1+1))/(a1+1)
        if a2 == -1:
            s2 = np.exp(p2) * (x2 - x1)
        else:
            s2 = np.exp(p2) * (np.exp(x2)**(a2+1) - np.exp(x1)**(a2+1))/(a2+1)
        s3 = np.exp(p3) * (0 - np.exp(x2)**(a3+1))/(a3+1)
        norm = np.log(s1+s2+s3)
        return np.exp(line-norm)
    else:
        return np.exp(line)

# copied from numpy 1.7 polynomial.py because numpy 1.3 on the LIGO clusters is too old and does not return cov matrix
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):

    import numpy.core.numeric as NX
    from numpy.core import isscalar, abs, dot
    from numpy.lib.twodim_base import diag, vander
    from numpy.linalg import eigvals, lstsq, inv
    try:
        from numpy.core import finfo # 1.7
    except:
        from numpy.lib.getlimits import finfo # 1.3 support for cluster

    order = int(deg) + 1
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0

    # check arguments.
    if deg < 0 :
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2 :
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0] :
        raise TypeError("expected x and y to have same length")

    # set rcond
    if rcond is None :
        rcond = len(x)*finfo(x.dtype).eps

    # set up least squares equation for powers of x
    lhs = vander(x, order)
    rhs = y

    # apply weighting
    if w is not None:
        w = NX.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError, "expected a 1-d array for weights"
        if w.shape[0] != y.shape[0] :
            raise TypeError, "expected w and y to have the same length"
        lhs *= w[:, NX.newaxis]
        if rhs.ndim == 2:
            rhs *= w[:, NX.newaxis]
        else:
            rhs *= w

    # scale lhs to improve condition number and solve
    scale = NX.sqrt((lhs*lhs).sum(axis=0))
    lhs /= scale
    c, resids, rank, s = lstsq(lhs, rhs, rcond)
    c = (c.T/scale).T  # broadcast scale coefficients

    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order and not full:
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    if full :
        return c, resids, rank, s, rcond
    elif cov :
        Vbase = inv(dot(lhs.T,lhs))
        Vbase /= NX.outer(scale, scale)
        # Some literature ignores the extra -2.0 factor in the denominator, but
        #  it is included here because the covariance of Multivariate Student-T
        #  (which is implied by a Bayesian uncertainty analysis) includes it.
        #  Plus, it gives a slightly more conservative estimate of uncertainty.
        fac = resids / (len(x) - order - 2.0)
        if y.ndim == 1:
            return c, Vbase * fac
        else:
            return c, Vbase[:,:,NX.newaxis] * fac
    else :
        return c

