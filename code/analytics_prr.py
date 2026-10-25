# Source: app/analytics/prr.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
Proportional Reporting Ratio (PRR) with chi-squared test.
"""
import numpy as np


def compute_prr(ct: dict) -> dict:
    """
    Compute PRR with 95% CI (Woolf method) and chi-squared statistic.

    PRR = (a/(a+b)) / (c/(c+d))
    where a=n11, b=n10, c=n01, d=n00

    Signal criteria: PRR ≥ 2 AND chi² ≥ 4
    """
    n11, n10, n01, n00 = ct["n11"], ct["n10"], ct["n01"], ct["n00"]

    if n11 == 0 or (n11 + n10) == 0 or (n01 + n00) == 0:
        return {"prr": None, "prr_ci_lower": None, "prr_chi2": None}

    try:
        a, b = n11, n10
        c, d = n01, n00

        if c == 0 or (b + d) == 0 or (a + c) == 0:
            return {"prr": None, "prr_ci_lower": None, "prr_chi2": None}

        prr = (a / (a + b)) / (c / (c + d))

        # 95% CI using Woolf method
        log_prr = np.log(prr)
        se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
        ci_lower = np.exp(log_prr - 1.96 * se)

        # Chi-squared with Yates' correction
        total = a + b + c + d
        denom = (a + b) * (c + d) * (a + c) * (b + d)
        chi2_stat = (abs(a * d - b * c) - total / 2) ** 2 * total / denom if denom > 0 else 0

        return {
            "prr": round(float(prr), 4),
            "prr_ci_lower": round(float(ci_lower), 4),
            "prr_chi2": round(float(chi2_stat), 4),
        }
    except (ZeroDivisionError, ValueError, OverflowError):
        return {"prr": None, "prr_ci_lower": None, "prr_chi2": None}
