import sys
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

LABELS = ["fake", "real"]  # fixed order so matrices from different files line up


def build_confusion_matrix(path: str, name:str):
    if not os.path.exists(path):
        print(f"ERROR: file not found -> {path}")
        return


    df = pd.read_csv(path)

    required_cols = {"actual_label", "final_prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: {path} is missing required column(s): {missing}")
        return

    n_total = len(df)
    is_error = df["final_prediction"] == "ERROR"
    n_errors = int(is_error.sum())

    valid = df.loc[~is_error].copy()
    n_valid = len(valid)

    y_true = valid["actual_label"]
    y_pred = valid["final_prediction"]

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {l}" for l in LABELS],
        columns=[f"Pred: {l}" for l in LABELS],
    )

    print(f"\n{'='*50}")
    print(f"{name}  ({path})")
    print(f"{'='*50}")
    print(f"Total rows: {n_total} | Excluded (ERROR): {n_errors} | Used: {n_valid}")
    print()
    print(cm_df.to_string())

    # save CSV
    cm_csv_path = f"confusion_matrix_{name}.csv"
    cm_df.to_csv(cm_csv_path)

    # save PNG heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([f"Pred: {l}" for l in LABELS])
    ax.set_yticklabels([f"Actual: {l}" for l in LABELS])
    ax.set_title(f"{name}\nConfusion Matrix (N={n_valid}, {n_errors} excluded)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14)
    fig.tight_layout()
    png_path = f"confusion_matrix_{name}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved: {cm_csv_path}, {png_path}")


build_confusion_matrix("/workspaces/Misinformation_Detector/Data_Folder_CSVs/eval_checkpoint_llama_3.3_70B.csv", "confusion_matrix_llama_3.3_70B")
build_confusion_matrix("/workspaces/Misinformation_Detector/Data_Folder_CSVs/eval_checkpoint_llama_8B_HF_INFERENCE.csv", "confusion_matrix_llama_3.1_8B")
build_confusion_matrix("/workspaces/Misinformation_Detector/Data_Folder_CSVs/eval_checkpoint_qwen_3_8B.csv", "confusion_matrix_qwen_3_8B")