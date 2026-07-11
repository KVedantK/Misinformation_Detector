import numpy as np
import joblib

FEATURES  = ["vreg", "eireg_anger", "eireg_fear", "eireg_joy", "eireg_sadness"]
MODEL_PATH = "models/affect_classifier.pkl"

pipe = joblib.load(MODEL_PATH)
def predict_from_affect(scores: dict) -> dict:
    """
    Input:  dict with keys vreg, eireg_anger, eireg_fear, eireg_joy, eireg_sadness
    Output: dict with affect_pred (fake/real) and affect_confidence (0-1)

    Example:
        scores = {"vreg": 0.39, "eireg_anger": 0.46,
                  "eireg_fear": 0.46, "eireg_joy": 0.20,
                  "eireg_sadness": 0.42}
        result = predict_from_affect(scores)
        # {"affect_pred": "fake", "affect_confidence": 0.61}
    """
    x     = np.array([[scores.get(f, 0.5) for f in FEATURES]])
    pred  = pipe.predict(x)[0]
    proba = pipe.predict_proba(x)[0]
    return {
        "affect_pred":       "fake" if pred == 1 else "real",
        "affect_confidence": round(float(proba.max()), 4),
    }


# if __name__ == "__main__":
#     sample = {
#         "vreg":          0.391,
#         "eireg_anger":   0.458,
#         "eireg_fear":    0.458,
#         "eireg_joy":     0.200,
#         "eireg_sadness": 0.421
#     }
#     print(f"Sample result: {predict_from_affect(sample)}")