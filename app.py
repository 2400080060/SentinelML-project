from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.reliability import data_quality_score, drift_report, reliability_score
from src.explainability import nearest_opposite_counterfactual, uncertainty_queue
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
st.set_page_config(page_title="SentinelML", page_icon="shield", layout="wide")
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    ood = joblib.load(MODEL_DIR / "ood_detector.joblib")
    scaler = joblib.load(MODEL_DIR / "reference_scaler.joblib")
    reference = joblib.load(MODEL_DIR / "reference_data.joblib")
    target = joblib.load(MODEL_DIR / "reference_target.joblib")
    features = joblib.load(MODEL_DIR / "feature_names.joblib")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    stats = json.loads((MODEL_DIR / "reference_stats.json").read_text())
    return model, ood, scaler, reference, target, features, metadata, stats
try:
    model, ood, scaler, reference, reference_target, features, metadata, stats = load_artifacts()
except FileNotFoundError:
    st.error("Artifacts are missing. Run `py src/train.py` first.")
    st.stop()
st.title("SentinelML")
st.subheader("ML Reliability and Self-Healing Platform")
st.caption("Prediction, monitoring, reliability, explanation, and active learning in one prototype.")
m = metadata["metrics"][metadata["best_model"]]
cols = st.columns(5)
for c, label, key in zip(cols, ["CV Accuracy", "Test Accuracy", "Test F1", "Test ROC-AUC", "CV F1"], ["cv_accuracy_mean", "test_accuracy", "test_f1", "test_roc_auc", "cv_f1_mean"]):
    c.metric(label, f"{m[key]:.3f}")
st.info(f"Candidate model: {metadata['best_model']} | Version: {metadata['version']} | Selection uses 5-fold CV; test data is a final audit.")
tab1, tab2, tab3, tab4 = st.tabs(["Reliability Check", "Drift Monitor", "Explainability", "Active Learning"])
with tab1:
    st.markdown("### Evaluate an incoming sample")
    medians = reference.median()
    row_values = medians.copy()
    input_cols = st.columns(2)
    for i, feature in enumerate(features[:8]):
        with input_cols[i % 2]:
            row_values[feature] = st.number_input(feature, value=float(medians[feature]), format="%.5f", key=f"input_{feature}")
    if st.button("Run reliability analysis", type="primary"):
        sample = pd.DataFrame([row_values], columns=features)
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        confidence = float(np.max(probabilities))
        quality, issues = data_quality_score(sample.iloc[0], stats)
        is_ood = bool(ood.predict(scaler.transform(sample))[0] == -1)
        report = drift_report(reference, pd.concat([reference.iloc[:40], sample], ignore_index=True))
        max_psi = float(report["psi"].max())
        score, decision = reliability_score(confidence, quality, max_psi, is_ood)
        mc = st.columns(5)
        mc[0].metric("Prediction", str(metadata["target_names"][prediction]).title())
        mc[1].metric("Confidence", f"{confidence:.1%}")
        mc[2].metric("Quality", f"{quality:.0f}/100")
        mc[3].metric("Max PSI", f"{max_psi:.3f}")
        mc[4].metric("Reliability", f"{score:.0f}/100")
        if decision == "TRUST": st.success(f"Decision: {decision}")
        elif decision == "REVIEW": st.warning(f"Decision: {decision}")
        else: st.error(f"Decision: {decision}")
        st.error("OOD signal detected.") if is_ood else st.success("No OOD signal detected.")
        if issues: st.warning("Quality findings: " + "; ".join(issues[:8]))
with tab2:
    st.markdown("### Population Stability Index")
    shift = st.slider("Simulated distribution shift", 0.0, 0.30, 0.08, 0.01)
    rng = np.random.default_rng(42)
    current = reference.copy()
    current[features[0]] = current[features[0]] * (1 + shift) + rng.normal(0, reference[features[0]].std() * 0.08, len(reference))
    report = drift_report(reference, current)
    a, b, c = st.columns(3)
    a.metric("Features", len(report)); b.metric("PSI >= 0.10", int((report.psi >= 0.10).sum())); c.metric("PSI >= 0.25", int((report.psi >= 0.25).sum()))
    st.dataframe(report.style.format({"psi": "{:.4f}"}), use_container_width=True, hide_index=True)
with tab3:
    st.markdown("### Counterfactual explanation")
    medians = reference.median(); exp_row = medians.copy()
    for feature in features[:8]:
        exp_row[feature] = st.number_input(f"Explanation - {feature}", value=float(medians[feature]), format="%.5f", key=f"exp_{feature}")
    if st.button("Generate counterfactual"):
        result = nearest_opposite_counterfactual(model, reference, reference_target, pd.DataFrame([exp_row], columns=features))
        if result is not None:
            st.success(f"Current prediction: {metadata['target_names'][result['prediction']]}")
            changes = result["changes"].head(10)
            st.dataframe(changes[["feature", "current", "counterfactual", "change"]].style.format({"current":"{:.4f}","counterfactual":"{:.4f}","change":"{:+.4f}"}), use_container_width=True, hide_index=True)
with tab4:
    st.markdown("### Active-learning queue")
    st.write("Lowest-confidence samples are prioritized for human labeling.")
    queue = uncertainty_queue(model, reference, top_k=12)
    st.dataframe(queue[["confidence", "uncertainty"] + features[:5]].style.format("{:.4f}"), use_container_width=True, hide_index=True)
with st.expander("Model benchmark"):
    st.dataframe(pd.DataFrame(metadata["metrics"]).T.style.format("{:.4f}"), use_container_width=True)
st.divider()
st.caption("Educational ML engineering prototype; benchmark data is medical-themed and not intended for clinical decisions.")
