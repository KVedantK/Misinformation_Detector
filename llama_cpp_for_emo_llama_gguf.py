from llama_cpp import Llama
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

llm = Llama(
    model_path="./emollama_gguf/Emollama-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1, 
    verbose=False
)


def prompt_vreg(text: str) -> str:
    """Sentiment strength — Vreg — best retrieval signal per paper"""
    return (
        f"Human:\nTask: Evaluate the valence intensity of the writer's mental state "
        f"based on the text, assigning it a real-valued score from 0 (most negative) "
        f"to 1 (most positive).\nText: {text}\nIntensity Score:\n\nA:\n>>"
    )

def prompt_voc(text: str) -> str:
    """Sentiment classification — Voc"""
    return (
        f"Human:\nTask: Categorize the text into an ordinal class that best characterizes "
        f"the writer's mental state, considering various degrees of positive and negative "
        f"sentiment intensity. 3: very positive mental state can be inferred. "
        f"2: moderately positive mental state can be inferred. "
        f"1: slightly positive mental state can be inferred. "
        f"0: neutral or mixed mental state can be inferred. "
        f"-1: slightly negative mental state can be inferred. "
        f"-2: moderately negative mental state can be inferred. "
        f"-3: very negative mental state can be inferred\n"
        f"Text: {text}\nIntensity Class:\n\nA:\n>>"
    )

def prompt_eireg(text: str, emotion: str) -> str:
    """Emotion intensity — EIreg — one call per emotion (anger/fear/joy/sadness)"""
    return (
        f"Human:\nTask: Assign a numerical value between 0 (least E) and 1 (most E) "
        f"to represent the intensity of emotion E expressed in the text.\n"
        f"Text: {text}\nEmotion: {emotion}\nIntensity Score:\n\nA:\n>>"
    )

def prompt_ec(text: str) -> str:
    """Emotion detection — Ec"""
    return (
        f"Human:\nTask: Categorize the text's emotional tone as either "
        f"'neutral or no emotion' or identify the presence of one or more of the given "
        f"emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, "
        f"sadness, surprise, trust).\nText: {text}\nThis text contains emotions:\n\nA:\n>>"
    )


def generate(prompt: str, max_tokens: int = 20) -> str:
    out = llm(prompt, max_tokens=max_tokens, temperature=0, echo=False)
    return out["choices"][0]["text"].strip()

def parse_float(raw: str, default: float = 0.5) -> float:
    """Extract first float found in output"""
    import re
    matches = re.findall(r"[-+]?\d*\.?\d+", raw)
    if matches:
        return max(0.0, min(1.0, float(matches[0])))
    return default

def parse_voc(raw: str) -> int:
    """Extract integer from -3 to 3"""
    import re
    matches = re.findall(r"-?\d+", raw)
    if matches:
        return max(-3, min(3, int(matches[0])))
    return 0

def parse_ec(raw: str) -> list:
    """Extract emotion labels from output"""
    emotions = ["anger","anticipation","disgust","fear","joy",
                "love","optimism","pessimism","sadness","surprise","trust"]
    return [e for e in emotions if e in raw.lower()]


def get_affective_profile(text: str) -> dict:
    """
    Returns all 5 affective dimensions for one article.
    This is what gets stored in your retrieval database.
    """
    text_short = text[:350]  

    # Vreg — sentiment strength (PRIMARY retrieval signal)
    vreg_raw = generate(prompt_vreg(text_short))
    vreg     = parse_float(vreg_raw)

    # Voc — sentiment class
    voc_raw  = generate(prompt_voc(text_short))
    voc      = parse_voc(voc_raw)

    # EIreg — emotion intensity x4
    eireg = {}
    for emotion in ["anger", "fear", "joy", "sadness"]:
        raw = generate(prompt_eireg(text_short, emotion))
        eireg[emotion] = parse_float(raw)

    # Ec — emotion detection
    ec_raw = generate(prompt_ec(text_short), max_tokens=30)
    ec     = parse_ec(ec_raw)

    return {
        "vreg":         vreg,           
        "voc":          voc,            
        "eireg_anger":  eireg["anger"], 
        "eireg_fear":   eireg["fear"],
        "eireg_joy":    eireg["joy"],
        "eireg_sadness":eireg["sadness"],
        "ec":           ec,             
    }


df = pd.read_csv("fakenewsnet_combined_cleaned.csv")

profiles = []
embeddings = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row["full_text"]

        profile = get_affective_profile(text)
        profiles.append(profile)

        if i % 50 == 0 and i > 0:
            pd.DataFrame(profiles).to_csv("profiles_checkpoint.csv", index=False)
        print(profile)
    except Exception as e:
        print("Exception : ", e)
        break
    
# Save final outputs
profiles_df = pd.DataFrame(profiles)
df_final = pd.concat([df.reset_index(drop=True), profiles_df], axis=1)
df_final.to_csv("ready_data.csv", index=False)

print(f"Done. Shape: {df_final.shape}")
print(df_final[["label", "vreg", "eireg_anger", "eireg_fear", "eireg_joy"]].groupby("label").mean())