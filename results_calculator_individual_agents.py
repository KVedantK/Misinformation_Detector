import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

LLAMA_FILE = "/workspaces/Misinformation_Detector/Data_Folder_CSVs/eval_checkpoint_llama.csv"
MISTRAL_FILE = "/workspaces/Misinformation_Detector/Data_Folder_CSVs/eval_checkpoint_mistral.csv"
LLAMA_3_8B_FILE = "/workspaces/Misinformation_Detector/Data_Folder_CSVs/eval_checkpoint_llama_8B_HF_INFERENCE.csv"


def evaluate_component(df, model_name, component, affect_weight, web_weight):


    p_fake = (
        affect_weight * df["p_fake_affect"] +
        web_weight * df["p_fake_web"]
    )


    prediction = (p_fake >= 0.5).map({
        True: "fake",
        False: "real"
    })

    actual = df["actual_label"].str.lower().str.strip()
    accuracy = accuracy_score(actual, prediction)

    precision = precision_score(
        actual,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        actual,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        actual,
        prediction,
        average="weighted",
        zero_division=0
    )

    return {
        "Model": model_name,
        "Component": component,
        "N": len(df),
        "Accuracy": accuracy,
        "Precision_w": precision,
        "Recall_w": recall,
        "F1_w": f1
    }



llama = pd.read_csv(LLAMA_FILE)
mistral = pd.read_csv(MISTRAL_FILE)
llama_3_8B = pd.read_csv(LLAMA_3_8B_FILE)



results = []

results.append(
    evaluate_component(
        llama_3_8B,
        "Llama-3.1 8B",
        "Affect-only",
        affect_weight=1.0,
        web_weight=0.0
    )
)
results.append(
    evaluate_component(
        llama_3_8B,
        "Llama-3.1 8B",
        "Web-only",
        affect_weight=0.0,
        web_weight=1.0
    )
)
results.append(
    evaluate_component(
        llama_3_8B,
        "Llama-3.1 8B",
        "Combined (0.4 affect + 0.6 web)",
        affect_weight=0.4,
        web_weight=0.6
    )
)


results.append(
    evaluate_component(
        llama,
        "Llama-3.3-70B",
        "Affect-only",
        affect_weight=1.0,
        web_weight=0.0
    )
)

results.append(
    evaluate_component(
        llama,
        "Llama-3.3-70B",
        "Web-only",
        affect_weight=0.0,
        web_weight=1.0
    )
)

results.append(
    evaluate_component(
        llama,
        "Llama-3.3-70B",
        "Combined (0.4 affect + 0.6 web)",
        affect_weight=0.4,
        web_weight=0.6
    )
)


results.append(
    evaluate_component(
        mistral,
        "Mistral (checkpoint 2)",
        "Affect-only",
        affect_weight=1.0,
        web_weight=0.0
    )
)

results.append(
    evaluate_component(
        mistral,
        "Mistral (checkpoint 2)",
        "Web-only",
        affect_weight=0.0,
        web_weight=1.0
    )
)

results.append(
    evaluate_component(
        mistral,
        "Mistral (checkpoint 2)",
        "Combined (0.4 affect + 0.6 web)",
        affect_weight=0.4,
        web_weight=0.6
    )
)



results_df = pd.DataFrame(results)


print("\nComponent Ablation Results")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.10f}"
    )
)


results_df.to_csv(
    "component_ablation_same_testset.csv",
    index=False
)

print("\nSaved to:")
print("component_ablation_same_testset.csv")