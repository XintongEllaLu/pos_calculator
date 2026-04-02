"""
=================================================================
Phase Transition Probability (PoS) Calculator
FILE 2: Statistical Analysis
=================================================================

WHAT THIS FILE DOES:
  Three levels of analysis, each building on the last:

  LEVEL 1 — Benchmark PoS Table
    Success rates by [phase × area × modality]
    with 95% Wilson confidence intervals.
    This is the core output investors use.

  LEVEL 2 — Factor Analysis
    Chi-square tests: which factors significantly
    predict success beyond phase alone?
    Logistic regression: independent effect of each factor.

  LEVEL 3 — Logistic Regression Model
    Predicts P(success) for any new trial given its features.
    Output: calibrated probability + comparison percentile.

OUTPUT FILES:
  data/results/benchmark_pos.csv      <- main reference table
  data/results/factor_effects.csv     <- regression coefficients
  data/results/chi2_tests.csv         <- significance tests
  data/results/model_coefficients.json <- for calculator UI

HOW TO RUN:
  pip install statsmodels scikit-learn
  python 02_analysis.py
=================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import json, os, warnings
warnings.filterwarnings("ignore")

CLEAN   = "data/clean"
RESULTS = "data/results"
os.makedirs(RESULTS, exist_ok=True)


# ── LOAD ──────────────────────────────────────────────────────
# ── LOAD ──────────────────────────────────────────────────────

def load():
    df  = pd.read_csv(f"{CLEAN}/trials_clean.csv")

    # OPTIONAL: load pos_table if exists (safe, won't break)
    pos_path = f"{CLEAN}/pos_table.csv"
    if os.path.exists(pos_path):
        pos = pd.read_csv(pos_path)
        print(f"[Load] {len(df):,} trials | {len(pos):,} PoS cells")
    else:
        pos = None
        print(f"[Load] {len(df):,} trials loaded (no pos_table)")

    # 🔥 CRITICAL FIX (does NOT break anything)
    if "phase_clean" not in df.columns:
        df["phase_clean"] = df["phase"]

    return df, pos
# ═════════════════════════════════════════════════════════════
# LEVEL 1 — BENCHMARK PoS TABLE
# ═════════════════════════════════════════════════════════════

def build_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    The primary output: a reference table that answers
    "what is the historical success rate for a trial like this?"

    Shows PoS stratified by phase, area, and modality.
    Includes sample size so users know how reliable each estimate is.
    """
    print("\n[Level 1] Building benchmark PoS table...")

    # Overall by phase (anchor numbers)
    overall = (
        df.groupby("phase_clean")["success"]
        .agg(n="count", n_success="sum")
        .assign(group="Overall")
        .reset_index()
        .rename(columns={"phase_clean": "phase"})
    )

    # By phase × area
    by_area = (
        df.groupby(["phase_clean", "area"])["success"]
        .agg(n="count", n_success="sum")
        .reset_index()
        .rename(columns={"phase_clean": "phase", "area": "group"})
    )

    # By phase × modality
    by_mod = (
        df.groupby(["phase_clean", "modality"])["success"]
        .agg(n="count", n_success="sum")
        .reset_index()
        .rename(columns={"phase_clean": "phase", "modality": "group"})
    )

    bench = pd.concat([overall, by_area, by_mod], ignore_index=True)
    bench["pos_pct"] = (bench["n_success"] / bench["n"]  * 100).round(1)

    # Wilson CI
    lo, hi = proportion_confint(bench["n_success"], bench["n"], method="wilson")
    bench["ci_lo"]    = (lo * 100).round(1)
    bench["ci_hi"]    = (hi * 100).round(1)
    bench["reliable"] = bench["n"] >= 30

    path = f"{RESULTS}/benchmark_pos.csv"
    bench.to_csv(path, index=False)
    print(f"  Saved -> {path}")

    # Print key numbers
    print("\n  === BENCHMARK PoS (industry average) ===")
    overall_summary = bench[bench["group"] == "Overall"][
        ["phase", "n", "pos_pct", "ci_lo", "ci_hi"]
    ]
    print(overall_summary.to_string(index=False))

    return bench


# ═════════════════════════════════════════════════════════════
# LEVEL 2 — CHI-SQUARE TESTS
# ═════════════════════════════════════════════════════════════

def run_chi2_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each factor, test whether success rates differ
    significantly across categories, controlling for phase.
    """
    print("\n[Level 2] Chi-square tests for each factor...")

    factors = ["area", "modality", "is_industry", "is_rct", "enroll_bucket"]
    results = []

    for phase in df["phase_clean"].unique():
        sub = df[df["phase_clean"] == phase]

        for factor in factors:
            try:
                ct = pd.crosstab(sub[factor], sub["success"])
                if ct.shape[0] < 2:
                    continue
                chi2, p, dof, _ = stats.chi2_contingency(ct)
                results.append({
                    "phase":      phase,
                    "factor":     factor,
                    "chi2":       round(chi2, 2),
                    "p_value":    round(p, 4),
                    "dof":        dof,
                    "n":          len(sub),
                    "significant": "Yes" if p < 0.05 else "No",
                })
            except Exception:
                pass

    result_df = pd.DataFrame(results).sort_values(["phase", "p_value"])
    path = f"{RESULTS}/chi2_tests.csv"
    result_df.to_csv(path, index=False)
    print(f"  Saved -> {path}")

    # Print summary
    sig = result_df[result_df["significant"] == "Yes"]
    print(f"\n  Significant factors (p<0.05): {len(sig)} / {len(result_df)}")
    for _, row in sig.iterrows():
        print(f"    {row['phase']:<15} {row['factor']:<20} p={row['p_value']:.4f}")

    return result_df


# ═════════════════════════════════════════════════════════════
# LEVEL 3 — LOGISTIC REGRESSION MODEL (FIXED VERSION)
# ═════════════════════════════════════════════════════════════

def build_model(df: pd.DataFrame) -> dict:
    """
    Logistic regression to estimate P(success) for a given trial.
    """

    print("\n[Level 3] Fitting logistic regression model...")

    # ── Feature engineering
    model_df = df.copy()

    model_df["log_enrollment"] = np.log1p(model_df["enrollment"])

    model_df["log_duration"] = np.log1p(
        model_df["planned_duration_months"].fillna(24)
    )

    # ── One-hot encode categoricals with explicit reference levels
    # Reference: phase="Phase 1", area="CNS", modality="Biologic"
    # We manually drop the reference columns after get_dummies
    cat_cols = ["phase_clean", "area", "modality"]

    encoded = pd.get_dummies(
        model_df[cat_cols],
        drop_first=False      # drop manually below so reference is predictable
    ).astype(float)

    for ref_col in ["phase_clean_Phase 1", "area_CNS", "modality_Biologic"]:
        if ref_col in encoded.columns:
            encoded = encoded.drop(columns=[ref_col])

    # ── Numeric features
    num_cols = ["is_industry", "is_rct", "log_enrollment", "log_duration"]

    X = pd.concat([
        encoded,
        model_df[num_cols].reset_index(drop=True)
    ], axis=1).fillna(0)

    y = model_df["success"]

    # ── Remove near-zero variance columns
    X = X.loc[:, X.std() > 0.01]
    feature_names = X.columns.tolist()

    # ── Scale continuous features and save params for the calculator
    scaler = StandardScaler()
    X_scaled = X.copy()

    scaler_params = {}
    for col in ["log_enrollment", "log_duration"]:
        if col in X_scaled.columns:
            vals = X_scaled[[col]]
            scaler_params[col] = {
                "mean": float(vals.mean().iloc[0]),
                "std":  float(vals.std().iloc[0]),
            }
            X_scaled[col] = scaler.fit_transform(vals)

    # ─────────────────────────────
    # 🔥 CRITICAL FIX BLOCK (dtype issue)
    # ─────────────────────────────

    # Force all numeric
    X_scaled = X_scaled.apply(pd.to_numeric, errors="coerce")

    # Drop rows with NA (safe)
    mask = X_scaled.notna().all(axis=1)
    X_scaled = X_scaled[mask]
    y = y[mask]

    # Add constant
    X_sm = sm.add_constant(X_scaled)

    # Force float dtype
    X_sm = X_sm.astype(float)
    y = y.astype(float)

    # ── Fit model
    logit = sm.Logit(y, X_sm).fit(disp=False)

    print(f"  Pseudo R²: {logit.prsquared:.3f}")
    print(f"  Features:  {len(feature_names)}")

    # ── Cross-validated AUC (sklearn)
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    auc_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc")

    print(f"  5-fold AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    # ── Save coefficients table
    coef_df = pd.DataFrame({
        "feature":    ["const"] + feature_names,
        "coef":       logit.params.values,
        "std_err":    logit.bse.values,
        "z":          logit.tvalues.values,
        "p_value":    logit.pvalues.values,
        "odds_ratio": np.exp(logit.params.values),
        "ci_lo":      np.exp(logit.conf_int().iloc[:, 0].values),
        "ci_hi":      np.exp(logit.conf_int().iloc[:, 1].values),
    })

    coef_df["significant"] = coef_df["p_value"].apply(
        lambda p: "Yes" if p < 0.05 else "No"
    )

    coef_path = f"{RESULTS}/factor_effects.csv"
    coef_df.to_csv(coef_path, index=False)

    print(f"  Saved coefficients -> {coef_path}")

    # ── Serialize model
    model_meta = {
        "auc":            round(float(auc_scores.mean()), 3),
        "pseudo_r2":      round(float(logit.prsquared), 3),
        "n_train":        int(len(y)),
        "feature_names":  feature_names,
        "intercept":      float(logit.params["const"]),
        "coefficients":   {
            k: float(v)
            for k, v in zip(feature_names, logit.params[1:])
        },
        "reference_levels": {
            "phase_clean": "Phase 1",
            "area":        "CNS",
            "modality":    "Biologic",
        },
        "scaler_params": scaler_params,
        # phase_pos is populated in main() after build_benchmark() runs
        "phase_pos": {},
    }

    meta_path = f"{RESULTS}/model_coefficients.json"

    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2)

    print(f"  Saved model meta -> {meta_path}")

    # ── Print top predictors
    sig_coefs = coef_df[
        (coef_df["significant"] == "Yes") &
        (coef_df["feature"] != "const")
    ].sort_values("coef", ascending=False)

    print("\n  Top positive predictors (increase PoS):")
    for _, r in sig_coefs.head(5).iterrows():
        print(f"    {r['feature']:<40} OR={r['odds_ratio']:.2f}  p={r['p_value']:.4f}")

    print("\n  Top negative predictors (decrease PoS):")
    for _, r in sig_coefs.tail(5).iterrows():
        print(f"    {r['feature']:<40} OR={r['odds_ratio']:.2f}  p={r['p_value']:.4f}")

    return model_meta


# ═════════════════════════════════════════════════════════════
# EXPORT DASHBOARD JSON
# ═════════════════════════════════════════════════════════════

def export_dashboard_json(df, bench, chi2, model_meta):
    """Package everything the HTML dashboard needs into one JSON."""

    # Summary stats
    summary = {
        "total_trials": int(len(df)),
        "success_rate": round(float(df["success"].mean() * 100), 1),
        "n_areas":      int(df["area"].nunique()),
        "n_modalities": int(df["modality"].nunique()),
        "date_range":   "2005-2025",
        "model_auc":    model_meta["auc"],
    }

    # PoS heatmap data: phase x area matrix
    hm = (
        df.groupby(["phase_clean", "area"])["success"]
        .agg(["count", "mean"])
        .reset_index()
    )
    hm.columns = ["phase_clean", "area", "n", "pos"]
    hm["pos"] = (hm["pos"] * 100).round(1)
    heatmap = hm[hm["n"] >= 30].to_dict(orient="records")

    # Benchmark bars (overall by phase)
    bars = bench[bench["group"] == "Overall"].to_dict(orient="records")

    # Factor breakdown: success rate per factor level within phase
    # Note: uses phase_clean column which matches "Phase 1" etc. after 01_clean fix
    factor_rows = []
    available_phases = df["phase_clean"].unique().tolist()
    for phase in ["Phase 1", "Phase 2", "Phase 3"]:
        if phase not in available_phases:
            continue
        sub = df[df["phase_clean"] == phase]
        for factor in ["area", "modality", "is_rct", "is_industry"]:
            if factor not in sub.columns:
                continue
            try:
                grp = (
                    sub.groupby(factor)["success"]
                    .agg(["count", "mean"])
                    .reset_index()
                )
                grp.columns = [factor, "n", "pos"]
                grp["pos"]    = (grp["pos"] * 100).round(1)
                grp["phase"]  = phase
                grp["factor"] = factor
                grp["level"]  = grp[factor].astype(str)
                grp = grp[grp["n"] >= 20][["phase", "factor", "level", "n", "pos"]]
                factor_rows.append(grp)
            except Exception:
                pass

    factors_df = pd.concat(factor_rows, ignore_index=True) if factor_rows else pd.DataFrame()

    payload = {
        "summary":        summary,
        "benchmark_bars": bars,
        "heatmap":        heatmap,
        "factor_data":    factors_df.to_dict(orient="records") if len(factors_df) else [],
        "model":          model_meta,
    }

    path = "data/dashboard_data.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Dashboard JSON -> {path}")
    return payload


def embed_in_html(payload):
    """
    Create report.html — a fully self-contained shareable file with JSON
    data embedded directly, so no server / fetch() is needed.
    """
    html_src = "index.html"
    html_out = "report.html"
    if not os.path.exists(html_src):
        print(f"  (skipping HTML embed — {html_src} not found)")
        return
    with open(html_src, "r", encoding="utf-8") as f:
        html = f.read()
    json_str = json.dumps(payload, default=str)
    html = html.replace(
        "window.EMBEDDED_DATA = null;",
        f"window.EMBEDDED_DATA = {json_str};",
    )
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Self-contained report  -> {html_out}  (share this file!)")


# ── MAIN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PoS Calculator — Step 2: Statistical Analysis")
    print("=" * 60)

    df, pos_table = load()

    bench      = build_benchmark(df)
    chi2       = run_chi2_tests(df)
    model_meta = build_model(df)

    # Attach phase-level PoS benchmarks so the HTML calculator can compare
    phase_bench = bench[bench["group"] == "Overall"]
    model_meta["phase_pos"] = {
        row["phase"]: {
            "pos":   float(row["pos_pct"]),
            "ci_lo": float(row["ci_lo"]),
            "ci_hi": float(row["ci_hi"]),
        }
        for _, row in phase_bench.iterrows()
    }

    # Also persist updated model_coefficients.json with phase_pos + scaler_params
    meta_path = f"{RESULTS}/model_coefficients.json"
    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2, default=str)

    payload = export_dashboard_json(df, bench, chi2, model_meta)
    embed_in_html(payload)

    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print("  Results saved in  data/results/")
    print("  Open report.html  (self-contained, shareable)")
    print("=" * 60)
