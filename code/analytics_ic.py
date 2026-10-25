# Source: app/analytics/ic.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
Information Component (IC) — WHO-UMC Bayesian shrinkage method.
"""
import numpy as np


def compute_ic(ct: dict) -> dict:
    """
    Compute IC with shrinkage.

    IC = log2((n11 + 0.5) / (expected + 0.5))
    expected = (n1x * nx1) / nxx

    Signal criteria: IC025 (lower 95% CI bound) > 0
    """
    n11, n1x, nx1, nxx = ct["n11"], ct["n1x"], ct["nx1"], ct["nxx"]

    if nxx == 0 or n1x == 0 or nx1 == 0:
        return {"ic": None, "ic025": None}

    expected = (n1x * nx1) / nxx

    # Shrinkage: add 0.5 to handle zeros
    ic = np.log2((n11 + 0.5) / (expected + 0.5))

    # Approximate 95% CI lower bound
    var_ic = 1 / ((n11 + 0.5) * np.log(2) ** 2)
    ic025 = ic - 3.3 * np.sqrt(var_ic)

    return {
        "ic": round(float(ic), 4),
        "ic025": round(float(ic025), 4),
    }
