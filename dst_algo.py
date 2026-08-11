import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)

# ─────────────────────────────────────────
# DST FUSION
# ─────────────────────────────────────────

AFFECT_RELIABILITY = 0.56   # logistic regression 5-fold CV F1

def dempster_shafer_fusion(p_fake: float, web_verdict: str,
                           web_confidence: float,
                           reliability: float = AFFECT_RELIABILITY) -> dict:
    p_real = 1.0 - p_fake

    # ── Agent 1 — affect mass ──────────────────────────────────
    m1_fake = p_fake * reliability
    m1_real = p_real * reliability
    m1_unc  = 1.0 - reliability

    # ── Agent 2 — web verdict mass ─────────────────────────────
    verdict_map = {
        "CONTRADICTED":          (0.90, 0.00),
        "SUPPORTED":             (0.00, 0.90),
        "INSUFFICIENT_EVIDENCE": (0.00, 0.00),
        "ERROR":                 (0.00, 0.00),
    }
    base_fake, base_real = verdict_map.get(str(web_verdict), (0.00, 0.00))
    m2_fake = base_fake * web_confidence
    m2_real = base_real * web_confidence
    m2_unc  = 1.0 - m2_fake - m2_real

    # ── Dempster combination ───────────────────────────────────
    num_fake = (m1_fake * m2_fake +
                m1_fake * m2_unc  +
                m1_unc  * m2_fake)

    num_real = (m1_real * m2_real +
                m1_real * m2_unc  +
                m1_unc  * m2_real)

    num_unc  =  m1_unc  * m2_unc

    K = m1_fake * m2_real + m1_real * m2_fake
    normaliser = 1.0 - K

    if normaliser < 1e-6:
        m_fake, m_real, m_unc = m2_fake, m2_real, m2_unc
    else:
        m_fake = num_fake / normaliser
        m_real = num_real / normaliser
        m_unc  = num_unc  / normaliser

    # ── Pignistic probability ──────────────────────────────────
    pign_fake = m_fake + m_unc / 2.0
    pign_real = m_real + m_unc / 2.0

    final_label = "fake" if pign_fake > pign_real else "real"
    final_conf  = round(max(pign_fake, pign_real), 4)

    return {
        "dst_prediction":  final_label,
        "dst_confidence":  final_conf,
        "dst_pign_fake":   round(pign_fake, 4),
        "dst_pign_real":   round(pign_real, 4),
        "dst_m_fake":      round(m_fake,    4),
        "dst_m_real":      round(m_real,    4),
        "dst_m_unc":       round(m_unc,     4),
        "dst_conflict_K":  round(K,         4),
    }

# ─────────────────────────────────────────
# LOAD CSV
# ─────────────────────────────────────────
df = pd.read_csv("eval_checkpoint_llama.csv")

# Drop ERROR rows
df_clean = df[~df["final_prediction"].isin(["ERROR"])].copy().reset_index(drop=True)
print(f"Loaded {len(df)} rows | Errors: {(df['final_prediction']=='ERROR').sum()} | Evaluating: {len(df_clean)}")

# ─────────────────────────────────────────
# APPLY DST TO EVERY ROW
# ─────────────────────────────────────────
dst_results = []
for _, row in df_clean.iterrows():
    result = dempster_shafer_fusion(
        p_fake         = float(row["p_fake"]),
        web_verdict    = str(row["web_verdict"]),
        web_confidence = float(row["web_confidence"]),
    )
    dst_results.append(result)

dst_df = pd.DataFrame(dst_results)
df_clean = pd.concat([df_clean.reset_index(drop=True), dst_df], axis=1)
df_clean["dst_correct"] = df_clean["dst_prediction"] == df_clean["actual_label"]

# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────
y_true     = (df_clean["actual_label"]     == "fake").astype(int)
y_old      = (df_clean["final_prediction"] == "fake").astype(int)
y_dst      = (df_clean["dst_prediction"]   == "fake").astype(int)

def metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred),                              4),
        "F1":        round(f1_score(y_true, y_pred, average="weighted"),                4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted",
                                           zero_division=0),                            4),
        "Recall":    round(recall_score(y_true, y_pred, average="weighted",
                                        zero_division=0),                               4),
        "Fake_F1":   round(f1_score(y_true, y_pred, pos_label=1, zero_division=0),     4),
        "Real_F1":   round(f1_score(y_true, y_pred, pos_label=0, zero_division=0),     4),
        "Fake_Rec":  round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "Real_Rec":  round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
    }

old_m = metrics(y_true, y_old)
dst_m = metrics(y_true, y_dst)

# ─────────────────────────────────────────
# PRINT COMPARISON TABLE
# ─────────────────────────────────────────
print("\n" + "="*65)
print("OVERALL COMPARISON — Weighted Average Fusion vs DST Fusion")
print("="*65)
print(f"{'Metric':<20} {'Old (weighted avg)':>20} {'DST fusion':>15} {'Delta':>10}")
print("-"*65)
for metric in ["Accuracy","F1","Precision","Recall","Fake_F1","Real_F1","Fake_Rec","Real_Rec"]:
    old_v = old_m[metric]
    dst_v = dst_m[metric]
    delta = dst_v - old_v
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    print(f"  {metric:<18} {old_v:>20.4f} {dst_v:>15.4f} {arrow}{abs(delta):>8.4f}")

# ─────────────────────────────────────────
# PER DOMAIN BREAKDOWN
# ─────────────────────────────────────────
print("\n" + "="*65)
print("PER DOMAIN — F1 Comparison")
print("="*65)
print(f"{'Domain':<12} {'N':>4} {'Old F1':>10} {'DST F1':>10} {'Delta':>10}")
print("-"*65)

domains = sorted(df_clean["filename"].str.extract(r'^([a-z]+)')[0].unique())
for d in domains:
    mask  = df_clean["filename"].str.startswith(d)
    sub   = df_clean[mask]
    yt    = (sub["actual_label"]     == "fake").astype(int)
    yo    = (sub["final_prediction"] == "fake").astype(int)
    yd    = (sub["dst_prediction"]   == "fake").astype(int)
    f1_o  = f1_score(yt, yo, average="weighted", zero_division=0)
    f1_d  = f1_score(yt, yd, average="weighted", zero_division=0)
    delta = f1_d - f1_o
    arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
    print(f"  {d:<10} {len(sub):>4} {f1_o:>10.4f} {f1_d:>10.4f} {arrow}{abs(delta):>8.4f}")

# ─────────────────────────────────────────
# CHANGED PREDICTIONS — what flipped
# ─────────────────────────────────────────
changed = df_clean[df_clean["final_prediction"] != df_clean["dst_prediction"]].copy()
print(f"\n{'='*65}")
print(f"PREDICTION CHANGES: {len(changed)} articles flipped")
print(f"{'='*65}")

if len(changed) > 0:
    old_right = changed[changed["correct"] == True]
    dst_right = changed[changed["dst_correct"] == True]
    print(f"  Old was right, DST wrong: {len(old_right)}")
    print(f"  DST was right, old wrong: {len(dst_right)}")

    print(f"\n  Sample flipped articles:")
    print(f"  {'Filename':<25} {'Actual':>8} {'Old':>8} {'DST':>8} {'p_fake':>8} {'web':>15} {'K':>8}")
    print(f"  {'-'*80}")
    for _, r in changed.head(15).iterrows():
        print(f"  {r['filename']:<25} {r['actual_label']:>8} "
              f"{r['final_prediction']:>8} {r['dst_prediction']:>8} "
              f"{r['p_fake']:>8.3f} {str(r['web_verdict']):>15} "
              f"{r['dst_conflict_K']:>8.3f}")

# ─────────────────────────────────────────
# CONFLICT ANALYSIS
# ─────────────────────────────────────────
print(f"\n{'='*65}")
print("CONFLICT ANALYSIS (DST K factor)")
print(f"{'='*65}")
print(f"  Mean K (all):         {df_clean['dst_conflict_K'].mean():.4f}")
print(f"  Mean K (correct DST): {df_clean[df_clean['dst_correct']]['dst_conflict_K'].mean():.4f}")
print(f"  Mean K (wrong DST):   {df_clean[~df_clean['dst_correct']]['dst_conflict_K'].mean():.4f}")
print(f"  High conflict (K>0.2): {(df_clean['dst_conflict_K']>0.2).sum()} articles")

# ─────────────────────────────────────────
# SAVE FULL RESULTS
# ─────────────────────────────────────────
df_clean.to_csv("evaluation_report_with_dst.csv", index=False)
print(f"\nSaved → evaluation_report_with_dst.csv")

# ─────────────────────────────────────────
# CLASSIFICATION REPORTS
# ─────────────────────────────────────────
print(f"\n{'='*65}")
print("OLD COMBINER — Classification Report")
print(f"{'='*65}")
print(classification_report(y_true, y_old, target_names=["real","fake"]))

print(f"\n{'='*65}")
print("DST FUSION — Classification Report")
print(f"{'='*65}")
print(classification_report(y_true, y_dst, target_names=["real","fake"]))