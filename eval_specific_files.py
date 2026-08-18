"""
rerun_specific_files.py

Reruns the evaluator on specific files only.
Provide the list of filenames to rerun at the bottom.
Output saved to rerun_results.csv — merge manually into main CSV.
"""

import os
import re
import json
import numpy as np
import joblib
from llama_cpp import Llama
from tavily import TavilyClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI

load_dotenv()

# ─────────────────────────────────────────
# CONFIG — update paths as needed
# ─────────────────────────────────────────
EMOLLAMA_PATH  = "../MSc_Project/emollama_gguf/Emollama-7b.Q4_K_M.gguf"
MODEL_PATH     = "models/affect_classifier.pkl"
DATASET_PATH   = "./training/training/fakeNewsDataset/"   # root folder with fake/ and legit/
OUTPUT_FILE    = "rerun_results.csv"
FEATURES       = ["vreg", "eireg_anger", "eireg_fear", "eireg_joy", "eireg_sadness"]

os.environ["TAVILY_API_KEY"]  = os.getenv("TAVILY_KEY")

# ─────────────────────────────────────────
# FILES TO RERUN — edit this list
# ─────────────────────────────────────────
FILES_TO_RERUN = [
    "biz13.legit.txt",
    "biz14.legit.txt",
    "biz15.legit.txt",
    "biz16.legit.txt",
    "biz17.legit.txt",
    "biz18.legit.txt",
    "biz19.legit.txt",
    "biz20.legit.txt",
    "biz21.legit.txt",
    "biz22.legit.txt",
    "biz23.legit.txt",
    "biz24.legit.txt",
    "biz25.legit.txt",
    "biz26.legit.txt",
    "biz27.legit.txt",
    "biz28.legit.txt",
    "biz29.legit.txt",
    "biz30.legit.txt",
    "biz31.legit.txt",
    # add more filenames here as needed
]

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
print("Loading EmoLLaMA...")
llm_emo = Llama(
    model_path=EMOLLAMA_PATH,
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

print("Loading logistic regression...")
pipe = joblib.load(MODEL_PATH)

print("Loading Tavily + LLM...")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

llm_web = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    openai_api_key=os.getenv("HF_TOKEN"),
    openai_api_base="https://router.huggingface.co/v1",
    temperature=0.01,
    max_tokens=800,
)
web_chain = llm_web | StrOutputParser()
print("All models loaded ✓\n")

# ─────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────
def _prompt_vreg(text):
    return (f"Human:\nTask: Evaluate the valence intensity of the writer's mental state "
            f"based on the text, assigning it a real-valued score from 0 (most negative) "
            f"to 1 (most positive).\nText: {text}\nIntensity Score:\n\nA:\n>>")

def _prompt_eireg(text, emotion):
    return (f"Human:\nTask: Assign a numerical value between 0 (least E) and 1 (most E) "
            f"to represent the intensity of emotion E expressed in the text.\n"
            f"Text: {text}\nEmotion: {emotion}\nIntensity Score:\n\nA:\n>>")

def _prompt_voc(text):
    return (f"Human:\nTask: Categorize the text into an ordinal class that best characterizes "
            f"the writer's mental state. 3: very positive. 2: moderately positive. "
            f"1: slightly positive. 0: neutral. -1: slightly negative. "
            f"-2: moderately negative. -3: very negative.\n"
            f"Text: {text}\nIntensity Class:\n\nA:\n>>")

def _prompt_ec(text):
    return (f"Human:\nTask: Categorize the text's emotional tone as either "
            f"'neutral or no emotion' or identify the presence of one or more of the given "
            f"emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, "
            f"sadness, surprise, trust).\nText: {text}\nThis text contains emotions:\n\nA:\n>>")

def _generate(prompt, max_tokens=20):
    for max_chars in [350, 250, 150, 80]:
        try:
            out = llm_emo(prompt, max_tokens=max_tokens, temperature=0, echo=False)
            return out["choices"][0]["text"].strip()
        except RuntimeError as e:
            if "llama_decode returned -3" in str(e) and "Text:" in prompt:
                parts  = prompt.split("Text:")
                prompt = "Text:".join(parts[:-1]) + "Text:" + parts[-1][:max_chars]
                continue
            raise e
    return ""

def _parse_float(raw, default=0.5):
    matches = re.findall(r"[-+]?\d*\.?\d+", raw)
    return max(0.0, min(1.0, float(matches[0]))) if matches else default

def _parse_voc(raw):
    matches = re.findall(r"-?\d+", raw)
    return max(-3, min(3, int(matches[0]))) if matches else 0

def _parse_ec(raw):
    emotions = ["anger","anticipation","disgust","fear","joy",
                "love","optimism","pessimism","sadness","surprise","trust"]
    return [e for e in emotions if e in raw.lower()]

# ─────────────────────────────────────────
# PIPELINE MODULES
# ─────────────────────────────────────────
def get_affective_scores(text):
    t = text[:350]
    try:    vreg = _parse_float(_generate(_prompt_vreg(t)))
    except: vreg = 0.5
    try:    voc  = _parse_voc(_generate(_prompt_voc(t)))
    except: voc  = 0
    eireg = {}
    for emotion in ["anger", "fear", "joy", "sadness"]:
        try:    eireg[emotion] = _parse_float(_generate(_prompt_eireg(t, emotion)))
        except: eireg[emotion] = 0.0
    try:    ec = _parse_ec(_generate(_prompt_ec(t), max_tokens=30))
    except: ec = []
    return {
        "vreg": vreg, "voc": voc,
        "eireg_anger": eireg["anger"], "eireg_fear": eireg["fear"],
        "eireg_joy": eireg["joy"], "eireg_sadness": eireg["sadness"],
        "ec": ec,
    }

def get_affective_prediction(scores):
    x     = np.array([[scores.get(f, 0.5) for f in FEATURES]])
    pred  = pipe.predict(x)[0]
    proba = pipe.predict_proba(x)[0]
    return {
        "affect_pred":       "fake" if pred == 1 else "real",
        "affect_confidence": round(float(proba.max()), 4),
        "p_fake":            round(float(proba[1]), 4),
        "p_real":            round(float(proba[0]), 4),
    }

_QUERY_SYSTEM = SystemMessage(content=(
    "You are a search query generator. Given a news article, generate ONE concise "
    "search query to find evidence to verify or debunk it. "
    "Output only the query — no explanation, no punctuation at the end."
))

_VERDICT_SYSTEM = SystemMessage(content=(
    "You are a fact-checking assistant. Given a news article and web evidence, "
    "decide if the evidence supports or contradicts the article.\n"
    "Respond ONLY with valid JSON:\n"
    '{"verdict": "SUPPORTED" or "CONTRADICTED", "confidence": <0.0-1.0>, '
    '"explanation": "<one sentence>", '
    '"supporting_sources": ["url"], "contradicting_sources": ["url"]}'
))

def get_web_verdict(text):
    query_raw = web_chain.invoke([
        _QUERY_SYSTEM,
        HumanMessage(content=f"News article:\n{text[:600]}")
    ])
    query = query_raw.strip().strip('"').strip("'")

    response = tavily_client.search(
        query=query, max_results=10,
        search_depth="advanced", include_answer=True
    )
    snippets = [
        f"Source: {r['url']}\nTitle: {r['title']}\nText: {r['content'][:600]}"
        for r in response.get("results", [])
    ]
    evidence_text = "\n\n---\n\n".join(snippets)
    tavily_answer = response.get("answer", "")
    sources       = [r["url"] for r in response.get("results", [])]

    raw = web_chain.invoke([
        _VERDICT_SYSTEM,
        HumanMessage(content=(
            f"News article:\n{text[:400]}\n\n"
            f"Tavily summary: {tavily_answer}\n\n"
            f"Evidence:\n{evidence_text}"
        ))
    ])

    verdict, confidence, explanation = "INSUFFICIENT_EVIDENCE", 0.5, ""
    supporting, contradicting = [], []
    try:
        start  = raw.find("{")
        end    = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        verdict       = parsed.get("verdict",              verdict)
        confidence    = float(parsed.get("confidence",     confidence))
        explanation   = parsed.get("explanation",          "")
        supporting    = parsed.get("supporting_sources",   [])
        contradicting = parsed.get("contradicting_sources",[])
    except Exception as e:
        print(f"  [Web] JSON parse warning: {e}")

    return {
        "web_verdict":           verdict,
        "web_confidence":        confidence,
        "web_explanation":       explanation,
        "web_query":             query,
        "web_sources":           sources,
        "supporting_sources":    supporting,
        "contradicting_sources": contradicting,
    }

def fuse_evidence(affect_result, web_result):
    p_fake_affect = affect_result["p_fake"]
    verdict_map = {
        "CONTRADICTED":          affect_result["p_fake"] * 0.3 + 0.7,
        "SUPPORTED":             affect_result["p_fake"] * 0.3 + 0.1,
        "INSUFFICIENT_EVIDENCE": 0.5,
    }
    p_fake_web = verdict_map.get(web_result["web_verdict"], 0.5)
    p_fake_web = min(1.0, max(0.0,
        p_fake_web * web_result["web_confidence"] +
        0.5 * (1 - web_result["web_confidence"])
    ))
    combined_p_fake = 0.4 * p_fake_affect + 0.6 * p_fake_web
    final_label     = "fake" if combined_p_fake > 0.5 else "real"
    final_conf      = round(max(combined_p_fake, 1 - combined_p_fake), 4)
    return {
        "final_prediction": final_label,
        "final_confidence": final_conf,
        "combined_p_fake":  round(combined_p_fake, 4),
        "p_fake_affect":    round(p_fake_affect,   4),
        "p_fake_web":       round(p_fake_web,       4),
    }

def evaluate_article(text):
    scores        = get_affective_scores(text)
    affect_result = get_affective_prediction(scores)
    web_result    = get_web_verdict(text)
    fusion        = fuse_evidence(affect_result, web_result)
    return {**scores, **affect_result, **web_result, **fusion}

# ─────────────────────────────────────────
# LOAD SPECIFIC FILES AND RUN
# ─────────────────────────────────────────
def load_file(filename):
    """Find file in fake/ or legit/ folder and return text + actual label."""
    for label, folder in [("fake", "fake"), ("real", "legit")]:
        path = os.path.join(DATASET_PATH, folder, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip(), label
    raise FileNotFoundError(f"{filename} not found in fake/ or legit/")

print(f"Files to rerun: {len(FILES_TO_RERUN)}")
results = []

for filename in tqdm(FILES_TO_RERUN, desc="Rerunning"):
    try:
        text, actual_label = load_file(filename)
        print(f"\nProcessing: {filename} (actual={actual_label})")

        result = evaluate_article(text)
        correct = result["final_prediction"] == actual_label

        row = {
            "filename":          filename,
            "actual_label":      actual_label,
            "final_prediction":  result["final_prediction"],
            "final_confidence":  result["final_confidence"],
            "correct":           correct,
            "vreg":              result["vreg"],
            "eireg_anger":       result["eireg_anger"],
            "eireg_fear":        result["eireg_fear"],
            "eireg_joy":         result["eireg_joy"],
            "eireg_sadness":     result["eireg_sadness"],
            "affect_pred":       result["affect_pred"],
            "affect_confidence": result["affect_confidence"],
            "p_fake":            result["p_fake"],
            "web_verdict":       result["web_verdict"],
            "web_confidence":    result["web_confidence"],
            "web_explanation":   result["web_explanation"],
            "web_query":         result["web_query"],
            "p_fake_affect":     result["p_fake_affect"],
            "p_fake_web":        result["p_fake_web"],
            "combined_p_fake":   result["combined_p_fake"],
        }
        print(f"  → {result['final_prediction']} (correct={correct})")

    except Exception as e:
        print(f"  ERROR: {e}")
        row = {
            "filename":         filename,
            "actual_label":     actual_label if "actual_label" in dir() else "unknown",
            "final_prediction": "ERROR",
            "final_confidence": 0.0,
            "correct":          False,
            "web_explanation":  str(e),
        }

    results.append(row)

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*50}")
print(f"Saved {len(results)} results → {OUTPUT_FILE}")
print(f"{'='*50}")
print(out_df[["filename", "actual_label", "final_prediction", "correct"]].to_string())