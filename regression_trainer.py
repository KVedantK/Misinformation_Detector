import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv("ready_data.csv")

FEATURES = ["vreg", "eireg_anger", "eireg_fear", "eireg_joy", "eireg_sadness"]
X = df[FEATURES].fillna(0.5).values
y = (df["label"] == "fake").astype(int).values

# ─────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────
pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
])

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(pipe, X, y, cv=cv,
                        scoring=["f1_weighted", "accuracy",
                                 "precision_weighted", "recall_weighted"])

print("=" * 50)
print("Logistic Regression — 5-Fold CV Results")
print("=" * 50)
print(f"  F1        {scores['test_f1_weighted'].mean():.4f} ± {scores['test_f1_weighted'].std():.4f}")
print(f"  Accuracy  {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
print(f"  Precision {scores['test_precision_weighted'].mean():.4f}")
print(f"  Recall    {scores['test_recall_weighted'].mean():.4f}")


pipe.fit(X, y)

print("\nFeature weights:")
coef = pipe.named_steps["clf"].coef_[0]
for feat, c in sorted(zip(FEATURES, coef), key=lambda x: abs(x[1]), reverse=True):
    direction = "→ FAKE" if c > 0 else "→ REAL"
    print(f"  {feat:<20} {c:>8.4f}  {direction}")

y_pred = pipe.predict(X)
print("\nClassification Report (full training set):")
print(classification_report(y, y_pred, target_names=["real", "fake"]))

os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/affect_classifier.pkl")
print("Saved → models/affect_classifier.pkl")