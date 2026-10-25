# Source: app/analytics/ror.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
Reporting Odds Ratio (ROR) with 95% confidence interval.
"""
import numpy as np


def compute_ror(ct: dict) -> dict:
    """
    Compute ROR with 95% CI.

    ROR = (a*d) / (b*c)

    Signal criteria: ROR ≥ 2 AND lower CI > 1
    """
    a, b, c, d = ct["n11"], ct["n10"], ct["n01"], ct["n00"]

    if a == 0 or b == 0 or c == 0 or d == 0:
        return {"ror": None, "ror_ci_lower": None, "ror_ci_upper": None}

    try:
        ror = (a * d) / (b * c)
        log_ror = np.log(ror)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ci_lower = np.exp(log_ror - 1.96 * se)
        ci_upper = np.exp(log_ror + 1.96 * se)

        return {
            "ror": round(float(ror), 4),
            "ror_ci_lower": round(float(ci_lower), 4),
            "ror_ci_upper": round(float(ci_upper), 4),
        }
    except (ZeroDivisionError, ValueError, OverflowError):
        return {"ror": None, "ror_ci_lower": None, "ror_ci_upper": None}
