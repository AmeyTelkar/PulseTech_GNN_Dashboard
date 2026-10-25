# Source: app/analytics/ebgm.py
# Deposited with PulseTech PLOS ONE package (V7)
# No credentials or hard-coded paths.

"""
Empirical Bayes Geometric Mean (EBGM) - DuMouchel (1999) Multi-item
Gamma Poisson Shrinker (MGPS).

Reference:
    DuMouchel W. Bayesian data mining in large frequency tables, with an
    application to the FDA spontaneous reporting system. Am Stat.
    1999;53(3):177-190.

Model. For a drug-ADR pair with observed count n (= n11) and expected count
E = n1x * nx1 / nxx, n ~ Poisson(lambda * E). The prior on lambda is
    w * Gamma(alpha1, beta1) + (1 - w) * Gamma(alpha2, beta2)   (rate form).
The posterior is a two-component Gamma mixture with shapes alpha_k + n, rates
beta_k + E and weights Q_k proportional to w_k times the negative-binomial
marginal likelihood of n under component k.

    EBGM = exp(E[log lambda | n])          (geometric mean of the posterior)
         = exp( sum_k Q_k * (digamma(alpha_k + n) - log(beta_k + E)) )
    EB05 = 5th percentile of the posterior mixture

The five prior hyperparameters are estimated from all pairs of an age band by
maximum likelihood (fit_prior). Only pairs with n >= min_n are available, so
the likelihood is truncated at min_n. compute_ebgm() without a fitted prior
falls back to DEFAULT_PRIOR, which is only a starting value.

Correction history: an earlier revision returned the posterior arithmetic
mean and always used DEFAULT_PRIOR. The EBGM values stored in S8 Table were
produced by neither revision and must be regenerated (see README.md).
"""
import numpy as np
from scipy import optimize, stats
from scipy.special import digamma, gammaln

# Starting values for the optimiser (alpha1, beta1, alpha2, beta2, w)
DEFAULT_PRIOR = (0.2, 0.1, 2.0, 4.0, 1.0 / 3.0)


def _log_nb(n, E, a, b):
    """log P(n | component) - negative binomial marginal of Poisson-Gamma."""
    return (gammaln(a + n) - gammaln(a) - gammaln(n + 1)
            + a * np.log(b / (b + E)) + n * np.log(E / (b + E)))


def _log_mix(n, E, prior):
    a1, b1, a2, b2, w = prior
    return np.logaddexp(np.log(w) + _log_nb(n, E, a1, b1),
                        np.log1p(-w) + _log_nb(n, E, a2, b2))


def fit_prior(n, E, min_n=3, start=DEFAULT_PRIOR):
    """Maximum-likelihood estimate of (alpha1, beta1, alpha2, beta2, w).

    n, E : arrays of observed and expected counts for every evaluated pair
    min_n: the minimum count used to select pairs (likelihood is truncated)
    """
    n = np.asarray(n, dtype=float)
    E = np.asarray(E, dtype=float)
    keep = (E > 0) & (n >= min_n)
    n, E = n[keep], E[keep]
    if len(n) < 10:
        return tuple(start)

    below = np.arange(min_n, dtype=float)

    def unpack(t):
        return (np.exp(t[0]), np.exp(t[1]), np.exp(t[2]), np.exp(t[3]), 1.0 / (1.0 + np.exp(-t[4])))

    def nll(t):
        prior = unpack(t)
        ll = _log_mix(n, E, prior)
        if min_n > 0:
            p_below = np.exp(_log_mix(below[None, :], E[:, None], prior)).sum(axis=1)
            ll = ll - np.log(np.clip(1.0 - p_below, 1e-300, None))
        val = -np.sum(ll)
        return val if np.isfinite(val) else 1e300

    a1, b1, a2, b2, w = start
    t0 = np.array([np.log(a1), np.log(b1), np.log(a2), np.log(b2), np.log(w / (1 - w))])
    res = optimize.minimize(nll, t0, method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    a1, b1, a2, b2, w = unpack(res.x)
    # convention: component 1 is the one with the smaller prior mean
    if a1 / b1 > a2 / b2:
        a1, b1, a2, b2, w = a2, b2, a1, b1, 1.0 - w
    return (float(a1), float(b1), float(a2), float(b2), float(w))


def posterior(n, E, prior=DEFAULT_PRIOR):
    """Posterior shapes, rates and weights for one pair."""
    a1, b1, a2, b2, w = prior
    l1 = np.log(w) + _log_nb(n, E, a1, b1)
    l2 = np.log1p(-w) + _log_nb(n, E, a2, b2)
    q1 = float(np.exp(l1 - np.logaddexp(l1, l2)))
    return (a1 + n, b1 + E, a2 + n, b2 + E, q1)


def ebgm_eb05(n, E, prior=DEFAULT_PRIOR, q=0.05):
    """Return (EBGM, EB_q) for observed n and expected E."""
    A1, B1, A2, B2, Q1 = posterior(n, E, prior)
    Q2 = 1.0 - Q1
    ebgm = float(np.exp(Q1 * (digamma(A1) - np.log(B1)) + Q2 * (digamma(A2) - np.log(B2))))
    d1 = stats.gamma(a=A1, scale=1.0 / B1)
    d2 = stats.gamma(a=A2, scale=1.0 / B2)
    # the mixture quantile lies between the two component quantiles
    lo, hi = min(d1.ppf(q), d2.ppf(q)), max(d1.ppf(q), d2.ppf(q))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if Q1 * d1.cdf(mid) + Q2 * d2.cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return ebgm, 0.5 * (lo + hi)


def compute_ebgm(ct: dict, prior=None) -> dict:
    """EBGM and EB05 for one 2x2 table.

    ct    : dict with keys n11, n1x, nx1, nxx
    prior : fitted prior from fit_prior(); DEFAULT_PRIOR if None
    Signal criterion used by PulseTech: EB05 > 2
    """
    n11, n1x, nx1, nxx = ct["n11"], ct["n1x"], ct["nx1"], ct["nxx"]
    if nxx == 0 or n1x == 0 or nx1 == 0:
        return {"ebgm": None, "eb05": None}
    E = (n1x * nx1) / nxx
    if E <= 0:
        return {"ebgm": None, "eb05": None}
    try:
        ebgm, eb05 = ebgm_eb05(float(n11), float(E), prior or DEFAULT_PRIOR)
        return {"ebgm": round(ebgm, 4), "eb05": round(eb05, 4)}
    except Exception:
        return {"ebgm": None, "eb05": None}
