"""
RAEmoLLM baseline — Template 1 (Vreg) and Template 2 (Vreg + explicit) only.
No affect classifier, no web verification, no fusion. Pure paper reproduction.
"""
import os, glob, re
import numpy as np, pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv

load_dotenv()


DATASET_PATH = "training/training/fakeNewsDataset"
SOURCE_CSV   = "./Data_Folder_CSVs/ready_data.csv"
SOURCE_EMB   = "./Data_Folder_CSVs/fakenewsnet_embeddings.npy"
TOP_K        = 4
TASK_PROMPT  = "Determine whether the target text is 0. Fake or 1. Legit."
OUT_CSV      = "raemollm_results.csv"

# ── models ──────────────────────────────────────────────
llm_emo = Llama(model_path="../MSc_Project/emollama_gguf/Emollama-7b.Q4_K_M.gguf",
                 n_ctx=2048, n_gpu_layers=-1, embedding=True, verbose=False)

llm_mistral = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    openai_api_key=os.getenv("HF_TOKEN"),
    openai_api_base="https://router.huggingface.co/v1",
    temperature=0.01,
    max_tokens=800,
)

# ── retrieval source pool ───────────────────────────────
src_df = pd.read_csv(SOURCE_CSV)
src_emb = np.load(SOURCE_EMB)
src_norm = src_emb / (np.linalg.norm(src_emb, axis=1, keepdims=True) + 1e-8)

def retrieve(text):
    emb = np.array(llm_emo.embed(text[:1000]), dtype=np.float32)
    if emb.ndim > 1: emb = emb.mean(axis=0)
    q = emb / (np.linalg.norm(emb) + 1e-8)
    idx = np.argsort(-(src_norm @ q))[:TOP_K]
    return src_df.iloc[idx]

# ── prompts (paper Sec 2.3, verbatim structure) ─────────
def lbl(l): return "0. Fake." if l == "fake" else "1. Legit."

def template1(text, examples):
    demos = "\n".join(f"Text: {r['full_text'][:500]}. The label of this text: {lbl(r['label'])}"
                       for _, r in examples.iterrows())
    return (f"Task: {TASK_PROMPT}\nTarget text: {text[:600]}\n"
            f"Here are a few examples: {demos}\n"
            f"According to the above information, the label of target text:")

def template2(text, vreg, examples):
    demos = "\n".join(f"Text: {r['full_text'][:500]}. Sentiment intensity: {r['vreg']:.3f}. "
                       f"The label of this text: {lbl(r['label'])}"
                       for _, r in examples.iterrows())
    return (f"Task: {TASK_PROMPT}\nTarget text: {text[:600]} Sentiment intensity: {vreg:.3f}.\n"
            f"Here are a few examples retrieved by sentiment intensity: {demos}\n"
            f"According to the above information, the label of target text:")

def ask(prompt):
    raw = llm_mistral.invoke([HumanMessage(content=prompt)]).content.strip()
    m = re.search(r"[01]", raw)
    return "fake" if (m and m.group() == "0") else "real"

def get_vreg(text):
    p = (f"Human:\nTask: Evaluate the valence intensity of the writer's mental state "
         f"based on the text, assigning it a real-valued score from 0 (most negative) "
         f"to 1 (most positive).\nText: {text[:350]}\nIntensity Score:\n\nA:\n>>")
    out = llm_emo(p, max_tokens=20, temperature=0, echo=False)["choices"][0]["text"]
    m = re.findall(r"[-+]?\d*\.?\d+", out)
    return max(0.0, min(1.0, float(m[0]))) if m else 0.5

# ── dataset loader ───────────────────────────────────────
def load_files(path):
    items = []
    for label, folder in [("fake", "fake"), ("real", "legit")]:
        for fp in sorted(glob.glob(os.path.join(path, folder, "*.txt"))):
            with open(fp, encoding="utf-8", errors="ignore") as f:
                items.append({"filename": os.path.basename(fp), "actual_label": label, "text": f.read()})
    return items

# ── main loop ────────────────────────────────────────────
items = load_files(DATASET_PATH)
rows = []
for item in tqdm(items):
    try:
        examples = retrieve(item["text"])
        vreg = get_vreg(item["text"])
        p1 = ask(template1(item["text"], examples))
        p2 = ask(template2(item["text"], vreg, examples))
        rows.append({"filename": item["filename"], "actual_label": item["actual_label"],
                      "vreg": round(vreg, 4),
                      "template1_pred": p1, "template2_pred": p2,
                      "template1_correct": p1 == item["actual_label"],
                      "template2_correct": p2 == item["actual_label"]})
    except Exception as e:
        print(f"ERROR {item['filename']}: {e}")
        rows.append({"filename": item["filename"], "actual_label": item["actual_label"],
                      "template1_pred": "ERROR", "template2_pred": "ERROR"})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)  # checkpoint every item

df = pd.DataFrame(rows)
for tmpl in ["template1", "template2"]:
    clean = df[df[f"{tmpl}_pred"] != "ERROR"]
    y_true = (clean["actual_label"] == "fake").astype(int)
    y_pred = (clean[f"{tmpl}_pred"] == "fake").astype(int)
    print(f"[{tmpl}] N={len(clean)} "
          f"Acc={accuracy_score(y_true,y_pred):.4f} "
          f"F1={f1_score(y_true,y_pred,average='weighted'):.4f} "
          f"P={precision_score(y_true,y_pred,average='weighted'):.4f} "
          f"R={recall_score(y_true,y_pred,average='weighted'):.4f}")