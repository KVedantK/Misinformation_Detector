import sys, re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

IN_CSV = "/workspaces/Misinformation_Detector/Data_Folder_CSVs/eval_checkpoint_qwen_3_8B.csv"
MODEL_NAME = "Qwen 3 8B"

# filename prefix -> domain name (extend if you have other prefixes)
DOMAIN_MAP = {
    "biz": "Business", "edu": "Education", "entmt": "Entertainment",
    "polit": "Politics", "sports": "Sports", "tech": "Technology",
}

def get_domain(filename):
    m = re.match(r"([a-zA-Z]+)\d+", str(filename))
    prefix = m.group(1).lower() if m else "unknown"
    return DOMAIN_MAP.get(prefix, prefix)

df = pd.read_csv(IN_CSV)
df["domain"] = df["filename"].apply(get_domain)
df = df[df["final_prediction"].isin(["fake", "real"])]  # drop ERROR/unparsed rows

rows = []
for domain, g in df.groupby("domain"):
    rows.append({
        "Model": MODEL_NAME,
        "Domain": domain,
        "N": len(g),
        "Acc": accuracy_score(g["actual_label"], g["final_prediction"]),
        "F1": f1_score(g["actual_label"], g["final_prediction"], average="weighted", zero_division=0),
    })

# overall row too
rows.append({
    "Model": MODEL_NAME, "Domain": "Overall", "N": len(df),
    "Acc": accuracy_score(df["actual_label"], df["final_prediction"]),
    "F1": f1_score(df["actual_label"], df["final_prediction"], average="weighted", zero_division=0),
})

order = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology", "Overall"]
out = pd.DataFrame(rows)
out["Domain"] = pd.Categorical(out["Domain"], categories=order + list(set(out["Domain"]) - set(order)))
out = out.sort_values("Domain")

out_path = f"domain_results_{re.sub(r'[^A-Za-z0-9]+', '_', MODEL_NAME)}.csv"
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
print(f"\nSaved -> {out_path}")