import numpy as np
import pandas as pd

def data_quality_score(row: pd.Series, reference_stats: dict):
    issues = []
    score = 100.0
    n = max(len(reference_stats), 1)
    score -= float(row.isna().sum()) / n * 50.0
    for feature, stats in reference_stats.items():
        value = row.get(feature)
        if pd.isna(value):
            issues.append(f"{feature}: missing")
            continue
        value = float(value)
        if not np.isfinite(value):
            score -= 5.0
            issues.append(f"{feature}: non-finite")
        elif value < stats["min"] or value > stats["max"]:
            score -= 1.5
            issues.append(f"{feature}: outside reference range")
    return float(np.clip(score, 0, 100)), issues

def psi(reference, current, bins=10):
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def drift_report(reference_df, current_df):
    rows = []
    for feature in reference_df.columns:
        value = psi(reference_df[feature], current_df[feature])
        status = "Stable" if value < 0.10 else ("Watch" if value < 0.25 else "Drift")
        rows.append({"feature": feature, "psi": value, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)

def reliability_score(confidence, quality, max_psi, ood):
    confidence_component = float(confidence) * 100.0
    drift_component = max(0.0, 100.0 - min(float(max_psi) / 0.25, 1.0) * 100.0)
    score = 0.45 * confidence_component + 0.35 * float(quality) + 0.20 * drift_component
    if ood:
        score -= 30.0
    score = float(np.clip(score, 0, 100))
    if score >= 80:
        decision = "TRUST"
    elif score >= 60:
        decision = "REVIEW"
    else:
        decision = "LOW TRUST"
    return score, decision
