import numpy as np
import pandas as pd
from src.reliability import psi, reliability_score, drift_report

def test_psi_identical_distribution_is_near_zero():
    x = np.arange(100, dtype=float)
    assert psi(x, x) < 1e-8

def test_reliability_penalizes_ood():
    good, _ = reliability_score(0.95, 98, 0.02, False)
    bad, _ = reliability_score(0.95, 98, 0.02, True)
    assert bad < good

def test_drift_report_has_expected_columns():
    df = pd.DataFrame({"a": np.arange(50), "b": np.arange(50) * 2})
    report = drift_report(df, df)
    assert {"feature", "psi", "status"} <= set(report.columns)
    assert len(report) == 2
