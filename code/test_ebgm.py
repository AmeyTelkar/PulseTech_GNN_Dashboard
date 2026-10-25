"""
Self-test for analytics_ebgm.py (no database needed).

1. EBGM and EB05 agree with brute-force numerical integration of the
   posterior for several (n, E, prior) cases.
2. Basic MGPS behaviour: EBGM is shrunk towards the prior for small counts,
   tracks n/E for large counts, and exceeds 1 for strongly disproportionate
   pairs (the defect in the stored S8 values).
3. fit_prior() recovers a known prior from simulated data well enough that the
   EBGM values it implies are close to those of the true prior.

Usage:  python code/test_ebgm.py      (needs numpy and scipy)
Exit status 0 if every check passes.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytics_ebgm import DEFAULT_PRIOR, compute_ebgm, ebgm_eb05, fit_prior  # noqa: E402

failures = 0


def check(label, ok):
    global failures
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def brute_force(n, E, prior):
    """Posterior by direct integration: prior density x Poisson likelihood."""
    a1, b1, a2, b2, w = prior
    dens = lambda lam: (w * stats.gamma.pdf(lam, a1, scale=1 / b1)
                        + (1 - w) * stats.gamma.pdf(lam, a2, scale=1 / b2)) * stats.poisson.pmf(n, lam * E)
    hi = max(50.0, 20 * (n + 1) / E)
    pts = [n / E] if n > 0 else None
    Z = integrate.quad(dens, 0, hi, points=pts, limit=400)[0]
    elog = integrate.quad(lambda l: np.log(l) * dens(l), 0, hi, points=pts, limit=400)[0] / Z
    cdf = lambda x: integrate.quad(dens, 0, x, points=[p for p in (pts or []) if p < x] or None, limit=400)[0] / Z
    lo, up = 1e-9, hi
    for _ in range(60):
        mid = 0.5 * (lo + up)
        lo, up = (mid, up) if cdf(mid) < 0.05 else (lo, mid)
    return float(np.exp(elog)), 0.5 * (lo + up)


print("1. Formula against numerical integration")
priors = [DEFAULT_PRIOR, (0.2041, 0.05816, 1.415, 1.838, 0.0969)]
cases = [(3, 0.5), (3, 5.0), (10, 1.0), (50, 20.0), (8, 4.0), (120, 3.0)]
for prior in priors:
    for n, E in cases:
        got = ebgm_eb05(n, E, prior)
        ref = brute_force(n, E, prior)
        ok = abs(got[0] - ref[0]) / ref[0] < 1e-3 and abs(got[1] - ref[1]) / ref[1] < 2e-3
        check(f"n={n:>3}, E={E:>5}: EBGM {got[0]:.4f} vs {ref[0]:.4f}; EB05 {got[1]:.4f} vs {ref[1]:.4f}", ok)

print("\n2. MGPS behaviour")
big = compute_ebgm({"n11": 667, "n1x": 1000, "nx1": 1000, "nxx": 4520})   # E = 221, n/E = 3.0
check(f"strong pair (n=667, E=221, n/E=3.0): EBGM {big['ebgm']} > 1 and close to n/E", 2.8 < big["ebgm"] < 3.1)
null = ebgm_eb05(20, 20.0, DEFAULT_PRIOR)[0]
check(f"null pair (n=E=20): EBGM {null:.3f} shrunk below n/E = 1", null < 1.0)
small = ebgm_eb05(3, 0.01, DEFAULT_PRIOR)[0]
check(f"tiny count (n=3, E=0.01, n/E=300): EBGM {small:.1f} shrunk well below 300", small < 100)
e1, e2 = ebgm_eb05(30, 5.0)[0], ebgm_eb05(60, 5.0)[0]
check(f"monotone in n: EBGM(30, 5) = {e1:.2f} < EBGM(60, 5) = {e2:.2f}", e1 < e2)
eb = ebgm_eb05(30, 5.0)
check(f"EB05 {eb[1]:.3f} < EBGM {eb[0]:.3f}", eb[1] < eb[0])

print("\n3. Prior estimation on simulated data")
rng = np.random.default_rng(2026)
true = (0.3, 0.2, 3.0, 3.5, 0.25)
m = 20000
E = np.exp(rng.normal(0.5, 1.2, m))
comp = rng.random(m) < true[4]
lam = np.where(comp, rng.gamma(true[0], 1 / true[1], m), rng.gamma(true[2], 1 / true[3], m))
n = rng.poisson(lam * E)
keep = n >= 3
fit = fit_prior(n[keep], E[keep], min_n=3)
print("     true prior:", true)
print("     fitted    :", tuple(round(v, 3) for v in fit))
grid = [(3, 0.5), (5, 2.0), (10, 1.0), (40, 10.0), (100, 20.0)]
diffs = [abs(ebgm_eb05(a, b, fit)[0] / ebgm_eb05(a, b, true)[0] - 1) for a, b in grid]
check(f"EBGM under fitted vs true prior: max relative difference {max(diffs):.3f} < 0.10", max(diffs) < 0.10)

print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
sys.exit(1 if failures else 0)
