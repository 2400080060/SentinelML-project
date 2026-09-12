import pandas as pd

def nearest_opposite_counterfactual(model, reference_X, reference_y, sample):
    pred = int(model.predict(sample)[0])
    candidates = reference_X.loc[reference_y.to_numpy() != pred].copy()
    if candidates.empty:
        return None
    combined = pd.concat([reference_X, sample], ignore_index=True)
    std = combined.std().replace(0, 1.0)
    scaled_candidates = (candidates - reference_X.mean()) / std
    scaled_sample = (sample.iloc[0] - reference_X.mean()) / std
    distances = ((scaled_candidates - scaled_sample) ** 2).sum(axis=1)
    idx = distances.idxmin()
    cf = candidates.loc[idx]
    changes = []
    for feature in reference_X.columns:
        before = float(sample.iloc[0][feature])
        after = float(cf[feature])
        delta = after - before
        if abs(delta) > 1e-9:
            changes.append({"feature": feature, "current": before, "counterfactual": after, "change": delta, "abs_change": abs(delta)})
    return {"prediction": pred, "opposite_example": cf, "changes": pd.DataFrame(changes).sort_values("abs_change", ascending=False)}

def uncertainty_queue(model, X, top_k=10):
    probabilities = model.predict_proba(X)
    confidence = probabilities.max(axis=1)
    result = X.copy()
    result["confidence"] = confidence
    result["uncertainty"] = 1.0 - confidence
    return result.sort_values("uncertainty", ascending=False).head(top_k)
