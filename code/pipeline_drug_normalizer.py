# Source: app/pipeline/drug_normalizer.py
# Deposited with PulseTech PLOS ONE package (V6)
# Credentials and hard-coded paths have been replaced with environment variables.

"""
Drug Name Normalization — RxNorm API + rapidfuzz fuzzy matching.
Route normalization — map 'Unknown'/'Other' to NULL, normalize variants.
"""
import asyncio
import pandas as pd
import httpx
from rapidfuzz import process, fuzz

RXNORM_API = "https://rxnav.nlm.nih.gov/REST"

# In-memory cache; persisted to drug_norm_cache table after pipeline completes
CACHE: dict[str, tuple[str, str]] = {}

# Route values that should be treated as NULL (not real routes)
ROUTE_NULL_VALUES = {'unknown', 'not specified', 'n/a', 'na', '', 'other', 'various'}

# Route normalization map
ROUTE_NORM_MAP = {
    'intravenous use': 'Intravenous',
    'intravenous drip': 'Intravenous',
    'intravenous (not otherwise specified)': 'Intravenous',
    'intravenous': 'Intravenous',
    'oral use': 'Oral',
    'oral': 'Oral',
    'subcutaneous': 'Subcutaneous',
    'subcutaneous use': 'Subcutaneous',
    'intramuscular': 'Intramuscular',
    'intramuscular use': 'Intramuscular',
    'topical': 'Topical',
    'topical use': 'Topical',
    'cutaneous': 'Topical',
    'transdermal': 'Transdermal',
    'nasal': 'Nasal',
    'inhalation': 'Inhalation',
    'respiratory (inhalation)': 'Inhalation',
    'ophthalmic': 'Ophthalmic',
    'rectal': 'Rectal',
    'intrathecal': 'Intrathecal',
    'epidural': 'Epidural',
}


async def rxnorm_lookup(name: str, client: httpx.AsyncClient) -> tuple[str, str] | None:
    """Returns (normalized_name, rxcui) or None."""
    try:
        r = await client.get(
            f"{RXNORM_API}/rxcui.json",
            params={"name": name, "search": 1},
            timeout=5.0,
        )
        data = r.json()
        rxcui = data.get("idGroup", {}).get("rxnormId", [None])[0]
        if rxcui:
            r2 = await client.get(
                f"{RXNORM_API}/rxcui/{rxcui}/property.json",
                params={"propName": "RxNorm Name"},
                timeout=5.0,
            )
            canon = (
                r2.json()
                .get("propConceptGroup", {})
                .get("propConcept", [{}])[0]
                .get("propValue")
            )
            return (canon or name, rxcui)
    except Exception:
        pass
    return None


def fuzzy_normalize(name: str, known_names: list[str], threshold: int = 85) -> str:
    """Fuzzy match against already-normalized names."""
    if not known_names:
        return name
    result = process.extractOne(name, known_names, scorer=fuzz.token_sort_ratio)
    if result is None:
        return name
    match, score, _ = result
    return match if score >= threshold else name


async def normalize_drugs(drug_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize drug names via RxNorm API lookup + fuzzy fallback.
    Adds drugname_normalized and rxcui columns.
    """
    drug_df = drug_df.copy()
    drug_df["drugname_clean"] = (
        drug_df["drugname"].fillna("UNKNOWN").str.lower().str.strip()
    )

    unique_names = drug_df["drugname_clean"].unique().tolist()

    results: dict[str, tuple[str, str]] = {}
    names_to_lookup = [n for n in unique_names if n not in CACHE]

    # Batch RxNorm lookups with concurrency limit
    if names_to_lookup:
        semaphore = asyncio.Semaphore(10)  # limit concurrent API calls

        async def limited_lookup(name, client):
            async with semaphore:
                return await rxnorm_lookup(name, client)

        async with httpx.AsyncClient() as client:
            tasks = [limited_lookup(n, client) for n in names_to_lookup]
            responses = await asyncio.gather(*tasks)

        for name, resp in zip(names_to_lookup, responses):
            if resp:
                CACHE[name] = resp
                results[name] = resp
            else:
                # Fuzzy fallback against already-cached normalized names
                cached_names = [v[0] for v in CACHE.values()] if CACHE else []
                norm = fuzzy_normalize(name, cached_names)
                CACHE[name] = (norm, "")
                results[name] = (norm, "")

    # Include already-cached results
    for n in unique_names:
        if n in CACHE:
            results[n] = CACHE[n]

    drug_df["drugname_normalized"] = drug_df["drugname_clean"].map(
        lambda x: results.get(x, (x, ""))[0]
    )
    drug_df["rxcui"] = drug_df["drugname_clean"].map(
        lambda x: results.get(x, ("", ""))[1] or None
    )
    return drug_df


def normalize_route(drug_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize route values:
    - 'Unknown', 'Other', etc → NULL
    - 'Intravenous use' → 'Intravenous', etc.
    """
    drug_df = drug_df.copy()
    route_raw = drug_df["route"].fillna("").str.lower().str.strip()

    # Set NULL-equivalent values to None
    is_null_route = route_raw.isin(ROUTE_NULL_VALUES)

    # Map known route variants
    route_mapped = route_raw.map(ROUTE_NORM_MAP)

    # Use mapped value if available, else keep original (cleaned), else None
    drug_df["route_normalized"] = route_mapped.fillna(
        route_raw.where(~is_null_route, other=None)
    )

    return drug_df
