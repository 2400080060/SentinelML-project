from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.data import load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
X, y, target_names = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, C=1.0, random_state=42))]),
    "RBF SVM": Pipeline([("scaler", StandardScaler()), ("model", SVC(C=2.0, kernel="rbf", probability=True, random_state=42))]),
    "Random Forest": RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42, class_weight="balanced"),
    "Extra Trees": ExtraTreesClassifier(n_estimators=500, max_features="sqrt", random_state=42, class_weight="balanced"),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results, trained = {}, {}
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)[:, 1]
    cv_acc = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="f1")
    results[name] = {"cv_accuracy_mean": float(cv_acc.mean()), "cv_accuracy_std": float(cv_acc.std()), "cv_f1_mean": float(cv_f1.mean()), "test_accuracy": float(accuracy_score(y_test, pred)), "test_precision": float(precision_score(y_test, pred)), "test_recall": float(recall_score(y_test, pred)), "test_f1": float(f1_score(y_test, pred)), "test_roc_auc": float(roc_auc_score(y_test, proba))}
    trained[name] = estimator
best_name = max(results, key=lambda n: (results[n]["cv_accuracy_mean"], results[n]["cv_f1_mean"], results[n]["test_roc_auc"]))
best_model = trained[best_name]
reference_scaler = StandardScaler().fit(X_train)
ood_detector = IsolationForest(n_estimators=300, contamination=0.05, random_state=42).fit(reference_scaler.transform(X_train))
reference_stats = {}
for feature in X_train.columns:
    s = X_train[feature].astype(float)
    reference_stats[feature] = {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max()), "median": float(s.median()), "q01": float(s.quantile(0.01)), "q99": float(s.quantile(0.99))}
for name, obj in [("best_model.joblib", best_model), ("ood_detector.joblib", ood_detector), ("reference_scaler.joblib", reference_scaler), ("reference_data.joblib", X_train.reset_index(drop=True)), ("reference_target.joblib", y_train.reset_index(drop=True)), ("test_data.joblib", X_test.reset_index(drop=True)), ("test_target.joblib", y_test.reset_index(drop=True)), ("feature_names.joblib", X.columns.tolist())]:
    joblib.dump(obj, MODEL_DIR / name)
metadata = {"project": "SentinelML", "version": "1.0.0", "trained_at_utc": datetime.now(timezone.utc).isoformat(), "best_model": best_name, "target_names": target_names, "n_features": int(X.shape[1]), "n_train": int(len(X_train)), "n_test": int(len(X_test)), "selection_rule": "highest 5-fold CV accuracy, tie-break by CV F1", "metrics": results}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
(MODEL_DIR / "reference_stats.json").write_text(json.dumps(reference_stats, indent=2))
print(f"Production candidate: {best_name}")
for name, metric in results.items(): print(f"{name}: CV accuracy={metric['cv_accuracy_mean']:.4f}, test accuracy={metric['test_accuracy']:.4f}, test ROC-AUC={metric['test_roc_auc']:.4f}")
print("Artifacts created successfully.")
