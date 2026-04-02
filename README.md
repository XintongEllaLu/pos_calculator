# Phase Transition Probability (PoS) Calculator

> **Portfolio Project** · Lucy · 2024–2025  
> Empirical success-rate benchmarks for clinical trials, built from 480,000+ AACT records

---

## What this project does

Biotech investors use **Probability of Success (PoS)** as a core input to rNPV models.
The standard approach is to plug in industry-average benchmarks from Biomedtracker or BioMedTracker
(e.g. "Phase 2 Oncology = 33%") — but these are blunt instruments.

This project builds a **data-driven, asset-specific PoS estimator** that adjusts for:

- **Phase** (1 / 2 / 3 and combinations)
- **Therapeutic area** (9 categories: Oncology, CNS, Rare Disease, Immunology…)
- **Drug modality** (Small Molecule / Biologic / Device)
- **Sponsor type** (Industry vs Academic)
- **Trial design** (RCT vs Single-Arm)
- **Enrollment size** (log-transformed)

---

## Why AACT is the right data source

| Property | AACT |
|----------|------|
| Source | All ClinicalTrials.gov registrations |
| Size | 480,000+ trials |
| Label quality | Structured `overall_status` field (no scraping needed) |
| Completeness | Full protocol + result data |
| Access | Free, no API key, daily updates |
| Format | Pipe-delimited flat files — works directly with pandas |

Unlike M&A data (which requires matching acquirer/target names across inconsistent sources),
AACT labels are **deterministic and structured**. `overall_status = Completed` is success.
`Terminated` or `Withdrawn` is failure. No ambiguity.

---

## File structure

```
pos-calculator/
│
├── 00_setup.py         # Install packages + download AACT data
├── 01_clean.py         # Filter, label, classify, feature-engineer
├── 02_analysis.py      # Benchmark PoS table, chi-square tests, logistic regression
├── index.html          # Interactive dashboard + calculator (open in browser)
│
├── data/
│   ├── raw/            # AACT flat files (from 00_setup.py)
│   ├── clean/          # trials_clean.csv, pos_table.csv
│   ├── results/        # benchmark_pos.csv, factor_effects.csv, model_coefficients.json
│   └── dashboard_data.json  # single JSON for the dashboard
│
├── requirements.txt
└── README.md
```

---

## How to run

```bash
# Step 0: Install + download AACT data (~5 min, 150 MB)
python 00_setup.py

# Step 1: Clean data and build PoS table (~3 min)
python 01_clean.py

# Step 2: Statistical analysis + export dashboard JSON (~5 min)
python 02_analysis.py

# Step 3: Open the dashboard
python -m http.server 8000
# → go to http://localhost:8000
```

> **The dashboard works without running the Python scripts.**
> It contains embedded placeholder data based on published benchmarks
> (MIT Lo Lab 2019, BioMedTracker 2024 industry averages).
> Run the Python scripts to replace it with your own AACT-derived estimates.

---

## Manual AACT download (if 00_setup.py fails)

1. Go to https://aact.ctti-clinicaltrials.org/downloads
2. Click the most recent **pipe-delimited monthly archive**
3. Unzip to `data/raw/`
4. You need: `studies.txt`, `interventions.txt`, `browse_conditions.txt`,
   `sponsors.txt`, `design_groups.txt`, `eligibilities.txt`

---

## Statistical methods

### Benchmark PoS Table
Simple proportion: `success_rate = n_completed / (n_completed + n_terminated + n_withdrawn)`  
Confidence intervals: Wilson score method (better than normal approximation for proportions)

### Chi-Square Tests
Tests whether success rate differs significantly across factor levels,
stratified by phase. P < 0.05 = the factor has a statistically significant
independent association with trial completion.

### Logistic Regression Model
```
log(P/(1-P)) = β₀ + β_phase + β_area + β_modality 
             + β_sponsor·is_industry + β_rct·is_rct 
             + β_enroll·log(enrollment) + β_duration·log(duration)
```
Trained on 2005–2022 trials with resolved status.
Evaluated with 5-fold cross-validation AUC.

### Calculator
The interactive calculator uses the logistic regression coefficients
to adjust a phase-level base rate based on the user's inputs.
Output: adjusted PoS + 95% confidence interval + percentile vs all trials.

---

## Key findings (from placeholder benchmarks)

| Phase | Industry PoS | 95% CI |
|-------|-------------|--------|
| Phase 1 | 66.5% | 66.0 – 67.1% |
| Phase 2 | 56.4% | 55.9 – 56.9% |
| Phase 3 | 70.2% | 69.4 – 71.0% |

**Oncology** has the lowest Phase 2 PoS (~50%), consistent with published literature.  
**RCT design** adds ~6 percentage points vs single-arm studies.  
**Rare Disease** outperforms the average by ~5 pp — likely due to smaller, more targeted populations.

---

## How to use this in an rNPV model

```
Standard rNPV:
  NPV = Σ [Cash_Flow_t × PoS_phase × discount_factor_t]
  
With this calculator:
  Replace generic PoS_phase (e.g. 33% for all Phase 2)
  with asset-specific PoS from the calculator output
  (e.g. 41% for Phase 2 Rare Disease Biologic RCT)

Impact:
  +8 pp PoS on a Phase 2 asset worth $500M at success
  → adds ~$40M to the risk-adjusted valuation
```

---

## Skills demonstrated

- **Statistical analysis**: Wilson CIs, chi-square tests, logistic regression with `statsmodels`
- **Data engineering**: AACT flat file processing, MeSH term classification, feature engineering
- **Domain knowledge**: Clinical development stages, PoS, rNPV framework
- **Visualisation**: Interactive D3.js dashboard with live calculator
- **Reproducibility**: Fully scripted pipeline from raw data to output
