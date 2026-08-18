import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#IN_CSV = "/workspaces/Misinformation_Detector/Data_Folder_CSVs/raemollm_results_3.1_8B_llama.csv"
#IN_CSV = "/workspaces/Misinformation_Detector/Data_Folder_CSVs/raemollm_results_llama_3-70B.csv"
IN_CSV = "/workspaces/Misinformation_Detector/Data_Folder_CSVs/raemollm_results_qwen_3_8B.csv"

df = pd.read_csv(IN_CSV)
pred_cols = [c for c in df.columns if c.endswith("_pred")]

rows = []
for col in pred_cols:
    clean = df[df[col].isin(["fake", "real"])]
    y_true, y_pred = clean["actual_label"], clean[col]
    rows.append({
        "Component": col.replace("_pred", ""),
        "N": len(clean),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_w": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall_w": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_w": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    })

out = pd.DataFrame(rows)
out.to_csv("raemollm_metrics_qwen_3_8B.csv", index=False)
print(out)