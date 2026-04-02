"""
=================================================================
Phase Transition Probability (PoS) Calculator
FILE 1: Data Cleaning & Feature Engineering
=================================================================

SCOPE:
  Interventional trials, Phase 1/2/3 (and combinations)
  registered 2005-2022 with resolved status (Completed/Terminated/Withdrawn)

OUTPUT:
  data/clean/trials_clean.csv   -- ~80k-120k rows
  data/clean/pos_table.csv      -- PoS by [phase x area x modality]

HOW TO RUN:
  python 01_clean.py
=================================================================
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

RAW   = "data/raw"
CLEAN = "data/clean"
os.makedirs(CLEAN, exist_ok=True)

# ── PHASE NAME MAP ────────────────────────────────────────────
# Normalise uppercase internal keys to display names used throughout
PHASE_NAMES = {
    "PHASE1":        "Phase 1",
    "PHASE2":        "Phase 2",
    "PHASE3":        "Phase 3",
    "PHASE1/PHASE2": "Phase 1/2",
    "PHASE2/PHASE3": "Phase 2/3",
}

# ── THERAPEUTIC AREA RULES ────────────────────────────────────
# Matched against concatenated lowercase MeSH terms per trial.
# Rules are ordered: first match wins (specific before broad).
AREA_RULES = [
    ("Oncology",          ["cancer", "tumor", "tumour", "neoplas", "carcinoma",
                           "lymphoma", "leukemia", "leukaemia", "melanoma",
                           "sarcoma", "glioma", "myeloma"]),
    ("CNS",               ["alzheimer", "parkinson", "epilep",
                           "multiple sclerosis", "brain disease",
                           "central nervous system disease", "schizophrenia",
                           "depressive disorder", "bipolar", "dementia",
                           "autism", "neurodegenerative", "mental disorder"]),
    ("Cardiovascular",    ["heart disease", "heart failure", "coronary",
                           "arrhythmia", "myocardial", "atherosclerosis",
                           "vascular disease", "hypertension"]),
    ("Infectious Disease",["infection", "bacterial disease", "virus disease",
                           "hiv infection", "malaria", "hepatitis",
                           "tuberculosis", "fungal infection", "parasitic",
                           "pneumonia"]),
    ("Metabolic",         ["diabetes", "obesity", "metabolic disease",
                           "nutritional and metabolic", "lipid metabolism",
                           "thyroid disease", "endocrine system disease",
                           "insulin"]),
    ("Immunology",        ["autoimmune", "rheumatoid arthritis",
                           "systemic lupus", "crohn disease",
                           "inflammatory bowel", "psoriasis",
                           "immune system disease", "ankylosing"]),
    ("Respiratory",       ["asthma", "pulmonary disease", "copd",
                           "lung disease", "respiratory tract disease"]),
    ("Rare Disease",      ["rare disease", "genetic disease, inborn",
                           "congenital, hereditary, and neonatal"]),
]


# ─────────────────────────────────────────────
# 1. LOAD (chunked for large files)
# ─────────────────────────────────────────────

def load_raw(filename, usecols=None):
    path = f"{RAW}/{filename}"
    print(f"  Loading {filename}...", end=" ", flush=True)
    chunks = []
    for chunk in pd.read_csv(
        path, sep="|", chunksize=100_000,
        low_memory=False, on_bad_lines="skip",
    ):
        chunk.columns = chunk.columns.str.strip().str.lower()
        if usecols:
            cols_lower = [c.lower() for c in usecols]
            chunk = chunk[[c for c in cols_lower if c in chunk.columns]]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"{len(df):,} rows")
    return df


def load_all():
    studies = load_raw("studies.txt", [
        "nct_id", "overall_status", "phase", "enrollment",
        "study_type", "start_date", "completion_date",
        "why_stopped", "number_of_arms",
    ])
    interventions = load_raw("interventions.txt",
                             ["nct_id", "intervention_type", "name"])
    conditions    = load_raw("browse_conditions.txt",
                             ["nct_id", "mesh_term"])
    sponsors      = load_raw("sponsors.txt",
                             ["nct_id", "agency_class", "lead_or_collaborator"])
    return studies, interventions, conditions, sponsors


# ─────────────────────────────────────────────
# 2. FILTER
# ─────────────────────────────────────────────

EXTERNAL_STOP_KEYWORDS = [
    "covid", "pandemic", "funding", "business", "strategic", "market",
]


def is_external_stop(reason):
    if pd.isna(reason):
        return False
    return any(kw in str(reason).lower() for kw in EXTERNAL_STOP_KEYWORDS)


def filter_studies(df):
    df["study_type"]     = df["study_type"].astype(str).str.upper()
    df["overall_status"] = df["overall_status"].astype(str).str.upper()
    # Normalise phase to key format (uppercase, no spaces)
    df["phase"] = df["phase"].astype(str).str.upper().str.replace(" ", "")

    df = df[df["study_type"] == "INTERVENTIONAL"]
    print(f"    After interventional filter : {len(df):,}")

    df = df[df["phase"].isin(PHASE_NAMES)]
    print(f"    After phase filter          : {len(df):,}")

    df = df[df["overall_status"].isin(["COMPLETED", "TERMINATED", "WITHDRAWN"])]
    print(f"    After status filter         : {len(df):,}")

    df = df[~df["why_stopped"].apply(is_external_stop)]
    print(f"    After external-stop filter  : {len(df):,}")

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df = df[df["enrollment"].notna() & (df["enrollment"] > 0)]
    print(f"    After enrollment filter     : {len(df):,}")

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df[
        (df["start_date"] >= "2005-01-01") &
        (df["start_date"] <= "2025-12-31")
    ]
    print(f"    After date filter (2005-2025): {len(df):,}")

    # Convert phase keys to display names ("PHASE1" -> "Phase 1")
    df["phase"] = df["phase"].map(PHASE_NAMES)

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. LABEL  (Completed = success)
# ─────────────────────────────────────────────

def create_label(df):
    df["success"] = (df["overall_status"] == "COMPLETED").astype(int)
    return df


# ─────────────────────────────────────────────
# 4. THERAPEUTIC AREA  (9 buckets via MeSH)
# ─────────────────────────────────────────────

def build_area_lookup(conditions_df):
    agg = (
        conditions_df
        .groupby("nct_id")["mesh_term"]
        .apply(lambda x: " | ".join(x.dropna().astype(str)).lower())
    )

    def classify(text):
        if pd.isna(text) or not text:
            return "Other"
        for area, keywords in AREA_RULES:
            if any(kw in text for kw in keywords):
                return area
        return "Other"

    return agg.apply(classify).rename("area")


# ─────────────────────────────────────────────
# 5. MODALITY  (Biologic / Small Molecule / Other)
# ─────────────────────────────────────────────

def build_modality_lookup(interventions_df):
    df = interventions_df.copy()
    df["itype"] = df["intervention_type"].fillna("").astype(str).str.upper()
    df["iname"] = df["name"].fillna("").astype(str).str.lower()

    def classify(row):
        itype = row["itype"]
        iname = row["iname"]
        if itype == "BIOLOGICAL":
            return "Biologic"
        if itype == "DRUG":
            return "Small Molecule"
        # Fallback: scan drug name
        if any(k in iname for k in ["antibody", " mab", "mrna", "vaccine",
                                     "gene therapy", "cell therapy", "protein"]):
            return "Biologic"
        if any(k in iname for k in ["tablet", "capsule", "inhibitor",
                                     "agonist", "antagonist", "compound"]):
            return "Small Molecule"
        return "Other"

    df["modality"] = df.apply(classify, axis=1)
    # Prefer most specific modality per trial (Biologic > Small Molecule > Other)
    priority = {"Biologic": 0, "Small Molecule": 1, "Other": 2}
    df["priority"] = df["modality"].map(priority)
    return (
        df.sort_values("priority")
        .groupby("nct_id")["modality"]
        .first()
    )


# ─────────────────────────────────────────────
# 6. SPONSOR TYPE
# ─────────────────────────────────────────────

def build_sponsor_lookup(sponsors_df):
    lead = sponsors_df[sponsors_df["lead_or_collaborator"] == "lead"].copy()
    lead["is_industry"] = (lead["agency_class"] == "Industry").astype(int)
    return lead.groupby("nct_id")["is_industry"].first()


# ─────────────────────────────────────────────
# 7. ASSEMBLE
# ─────────────────────────────────────────────

def assemble(studies, interventions, conditions, sponsors):
    print("\n[Assemble] Joining feature lookups...")

    df = studies.copy()
    df = df.join(build_area_lookup(conditions),        on="nct_id")
    df = df.join(build_modality_lookup(interventions), on="nct_id")
    df = df.join(build_sponsor_lookup(sponsors),       on="nct_id")

    df["area"]        = df["area"].fillna("Other")
    df["modality"]    = df["modality"].fillna("Other")
    df["is_industry"] = df["is_industry"].fillna(0).astype(int)

    df["number_of_arms"] = pd.to_numeric(df["number_of_arms"], errors="coerce").fillna(1)
    df["is_rct"] = (df["number_of_arms"] >= 2).astype(int)

    df["start_date"]      = pd.to_datetime(df["start_date"],      errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    df["planned_duration_months"] = (
        (df["completion_date"] - df["start_date"]).dt.days / 30.44
    ).clip(1, 240)

    keep = [
        "nct_id", "overall_status", "phase", "enrollment", "success",
        "area", "modality", "is_industry", "is_rct",
        "planned_duration_months", "start_date", "why_stopped",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


# ─────────────────────────────────────────────
# 8. PoS TABLE
# ─────────────────────────────────────────────

def build_pos_table(df):
    pos = (
        df.groupby(["phase", "area", "modality"])["success"]
        .agg(n="count", n_success="sum")
        .reset_index()
    )
    pos["pos_pct"] = (pos["n_success"] / pos["n"] * 100).round(1)
    return pos


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PoS Calculator — Step 1: Data Cleaning")
    print("=" * 60)
    print()

    studies, interventions, conditions, sponsors = load_all()

    print("\n[Filter] Applying inclusion/exclusion criteria...")
    studies = filter_studies(studies)
    studies = create_label(studies)

    df = assemble(studies, interventions, conditions, sponsors)

    print(f"\n[Summary]")
    print(f"  Final sample : {len(df):,} trials")
    print(f"  Success rate : {df['success'].mean()*100:.1f}%")
    print(f"\n  Therapeutic areas:")
    for area, cnt in df["area"].value_counts().items():
        print(f"    {area:<25} {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")
    print(f"\n  Modalities:")
    for mod, cnt in df["modality"].value_counts().items():
        print(f"    {mod:<25} {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    clean_path = f"{CLEAN}/trials_clean.csv"
    df.to_csv(clean_path, index=False)
    print(f"\n[OK] Saved clean trials -> {clean_path}")

    pos = build_pos_table(df)
    pos_path = f"{CLEAN}/pos_table.csv"
    pos.to_csv(pos_path, index=False)
    print(f"[OK] Saved PoS table    -> {pos_path}")

    print("\n  Next -> python 02_analysis.py")
