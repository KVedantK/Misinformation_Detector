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
from dotenv import load_dotenv
import glob
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
EMOLLAMA_PATH = "../MSc_Project/emollama_gguf/Emollama-7b.Q4_K_M.gguf"
MODEL_PATH    = "models/affect_classifier.pkl"
FEATURES      = ["vreg", "eireg_anger", "eireg_fear", "eireg_joy", "eireg_sadness"]

os.environ["HF_TOKEN"]                 = os.getenv("HF_TOKEN")
os.environ["TAVILY_API_KEY"]           = os.getenv("TAVILY_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["GROQ_API_KEY"]              = os.getenv("GROQ_KEY")

# ─────────────────────────────────────────
# LOAD MODELS (once at startup)
# ─────────────────────────────────────────
print("Loading EmoLLaMA...")
llm_emo = Llama(
    model_path=EMOLLAMA_PATH,
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

print("Loading logistic regression classifier...")
pipe = joblib.load(MODEL_PATH)

print("Loading Tavily + LLM chain...")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

## mistralai/mistral-small-3.2-24b-instruct

llm_web = ChatOpenAI(
    model="qwen/qwen3-8b",
    openai_api_key=os.getenv("OPEN_ROUTER_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.01,
    max_tokens=800,
)

## Llama 
# llm_web = ChatOpenAI(
#     model="meta-llama/llama-3.3-70b-instruct",
#     openai_api_key=os.getenv("OPEN_ROUTER_KEY"),
#     openai_api_base="https://openrouter.ai/api/v1",
#     temperature=0.01,
#     max_tokens=800,
# )


web_chain = llm_web | StrOutputParser()

print("All models loaded ✓\n")

# ═════════════════════════════════════════
# MODULE 1 — AFFECTIVE SCORING (EmoLLaMA)
# ═════════════════════════════════════════

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
    """Safe generation with progressive truncation on context overflow."""
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

def get_affective_scores(text: str) -> dict:
    """
    Module 1 — EmoLLaMA affective scoring.
    Returns 5 scores used by the logistic regression classifier.
    """
    t = text[:350]
    try:    vreg = _parse_float(_generate(_prompt_vreg(t)))
    except: vreg = 0.5

    try:    voc = _parse_voc(_generate(_prompt_voc(t)))
    except: voc = 0

    eireg = {}
    for emotion in ["anger", "fear", "joy", "sadness"]:
        try:    eireg[emotion] = _parse_float(_generate(_prompt_eireg(t, emotion)))
        except: eireg[emotion] = 0.0

    try:    ec = _parse_ec(_generate(_prompt_ec(t), max_tokens=30))
    except: ec = []

    return {
        "vreg":          vreg,
        "voc":           voc,
        "eireg_anger":   eireg["anger"],
        "eireg_fear":    eireg["fear"],
        "eireg_joy":     eireg["joy"],
        "eireg_sadness": eireg["sadness"],
        "ec":            ec,
    }

# ═════════════════════════════════════════
# MODULE 2 — LOGISTIC REGRESSION CLASSIFIER
# ═════════════════════════════════════════

def get_affective_prediction(scores: dict) -> dict:
    """
    Module 2 — Logistic regression on the 5 affective scores.
    Returns soft label (fake/real) + calibrated confidence.
    """
    x     = np.array([[scores.get(f, 0.5) for f in FEATURES]])
    pred  = pipe.predict(x)[0]
    proba = pipe.predict_proba(x)[0]
    return {
        "affect_pred":       "fake" if pred == 1 else "real",
        "affect_confidence": round(float(proba.max()), 4),
        "p_fake":            round(float(proba[1]), 4),
        "p_real":            round(float(proba[0]), 4),
    }

# ═════════════════════════════════════════
# MODULE 3 — WEB VERIFICATION (Tavily)
# ═════════════════════════════════════════

_QUERY_SYSTEM = SystemMessage(content=(
    "You are a search query generator. Given a news article, generate ONE concise "
    "search query for a search engine that would find evidence to verify or debunk it. "
    "Do not be too general — get specific information around the article. "
    "Output only the query — no explanation, no punctuation at the end."
))

_VERDICT_SYSTEM = SystemMessage(content=(
    "You are a fact-checking assistant. You will be given a news article and web "
    "evidence retrieved about it. Decide if the evidence supports or contradicts the article.\n"
    "Respond ONLY with valid JSON in exactly this format:\n"
    '{\n'
    '  "verdict": "SUPPORTED" or "CONTRADICTED",\n'
    '  "confidence": <number 0.0 to 1.0>,\n'
    '  "explanation": "<one sentence explaining your decision>",\n'
    '  "supporting_sources": ["<url>"],\n'
    '  "contradicting_sources": ["<url>"]\n'
    "}\n"
    "Output only the JSON. No preamble. No explanation outside the JSON."
))

def get_web_verdict(text: str) -> dict:
    """
    Module 3 — Two-chain web verification.
    Chain 1: LLM generates query → Tavily searches
    Chain 2: LLM reads evidence → JSON verdict
    """
    # Chain 1 — query generation + retrieval
    print("  [Web] Generating search query...")
    query_raw = web_chain.invoke([
        _QUERY_SYSTEM,
        HumanMessage(content=f"News article:\n{text[:600]}")
    ])
    query = query_raw.strip().strip('"').strip("'")
    print(f"  [Web] Query: {query}")

    response = tavily_client.search(
        query=query,
        max_results=10,
        search_depth="advanced",
        include_answer=True
    )
    snippets = [
        f"Source: {r['url']}\nTitle: {r['title']}\nText: {r['content'][:600]}"
        for r in response.get("results", [])
    ]
    evidence_text = "\n\n---\n\n".join(snippets)
    tavily_answer = response.get("answer", "")
    sources       = [r["url"] for r in response.get("results", [])]
    print(f"  [Web] Found {len(snippets)} sources")

    # Chain 2 — verdict generation
    print("  [Web] Generating verdict...")
    raw = web_chain.invoke([
        _VERDICT_SYSTEM,
        HumanMessage(content=(
            f"News article:\n{text[:400]}\n\n"
            f"Tavily summary: {tavily_answer}\n\n"
            f"Web evidence:\n{evidence_text}"
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

    print(f"  [Web] Verdict: {verdict} ({confidence:.2f})")

    return {
        "web_verdict":            verdict,
        "web_confidence":         confidence,
        "web_explanation":        explanation,
        "web_query":              query,
        "web_sources":            sources,
        "supporting_sources":     supporting,
        "contradicting_sources":  contradicting,
    }

# ═════════════════════════════════════════
# MODULE 4 — EVIDENCE FUSION
# ═════════════════════════════════════════

def fuse_evidence(affect_result: dict, web_result: dict) -> dict:
    """
    Module 4 — Combine affective confidence + web confidence
    into a single final prediction.

    Fusion logic:
      - Convert web verdict to a fake probability
          CONTRADICTED  → high fake probability (article contradicted by web)
          SUPPORTED     → low fake probability
          INSUFFICIENT  → neutral (0.5)
      - Weighted average: 40% affective + 60% web
          (web gets higher weight as it is grounded in live evidence)
      - Final threshold: > 0.5 → fake, <= 0.5 → real
    """
    p_fake_affect = affect_result["p_fake"]

    verdict_map = {
        "CONTRADICTED":          affect_result["p_fake"] * 0.3 + 0.7,
        "SUPPORTED":             affect_result["p_fake"] * 0.3 + 0.1,
        "INSUFFICIENT_EVIDENCE": 0.5,
    }
    p_fake_web = verdict_map.get(web_result["web_verdict"], 0.5)
    p_fake_web = min(1.0, max(0.0, p_fake_web * web_result["web_confidence"] +
                              0.5 * (1 - web_result["web_confidence"])))

    # Weighted fusion
    combined_p_fake = 0.4 * p_fake_affect + 0.6 * p_fake_web
    final_label     = "fake" if combined_p_fake > 0.5 else "real"
    final_conf      = round(max(combined_p_fake, 1 - combined_p_fake), 4)

    return {
        "final_prediction":  final_label,
        "final_confidence":  final_conf,
        "combined_p_fake":   round(combined_p_fake, 4),
        "p_fake_affect":     round(p_fake_affect, 4),
        "p_fake_web":        round(p_fake_web, 4),
    }

# ═════════════════════════════════════════
# FULL PIPELINE
# ═════════════════════════════════════════

def evaluate_article(text: str) -> dict:
    """
    Full pipeline for one article.
    Returns a dict with all intermediate and final results.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {text[:80].strip()}...")
    print(f"{'='*60}")

    # Step 1 — Affective scores
    print("\n[Step 1] Affective scoring...")
    scores = get_affective_scores(text)
    print(f"  vreg={scores['vreg']:.3f}  anger={scores['eireg_anger']:.3f}  "
          f"fear={scores['eireg_fear']:.3f}  joy={scores['eireg_joy']:.3f}  "
          f"sadness={scores['eireg_sadness']:.3f}")

    # Step 2 — Logistic regression
    print("\n[Step 2] Logistic regression classifier...")
    affect_result = get_affective_prediction(scores)
    print(f"  Affect prediction: {affect_result['affect_pred']} "
          f"(confidence={affect_result['affect_confidence']:.4f}  "
          f"p_fake={affect_result['p_fake']:.4f})")

    # Step 3 — Web verification
    print("\n[Step 3] Web verification...")
    web_result = get_web_verdict(text)

    # Step 4 — Fusion
    print("\n[Step 4] Evidence fusion...")
    fusion = fuse_evidence(affect_result, web_result)
    print(f"  p_fake (affect) = {fusion['p_fake_affect']:.4f}")
    print(f"  p_fake (web)    = {fusion['p_fake_web']:.4f}")
    print(f"  combined p_fake = {fusion['combined_p_fake']:.4f}")
    print(f"\n  ► FINAL: {fusion['final_prediction'].upper()} "
          f"(confidence={fusion['final_confidence']:.4f})")

    return {
        # Raw scores
        **scores,
        # Logistic regression output
        **affect_result,
        # Web verification output
        **web_result,
        # Fusion output
        **fusion,
    }

# ═════════════════════════════════════════
# MAIN — test on sample article
# ═════════════════════════════════════════
# ═════════════════════════════════════════
# BATCH EVALUATION — folder-based dataset
# fakeNewsDataset/
#   fake/   *.txt  → actual_label = fake
#   legit/  *.txt  → actual_label = real
# ═════════════════════════════════════════



DATASET_PATH  = "training/training/fakeNewsDataset/"          # folder containing fake/ and legit/
CHECKPOINT    = "eval_checkpoint_mistral.csv"
FINAL_REPORT  = "evaluation_report_mistral.csv"

def load_all_files(dataset_path: str) -> list[dict]:
    """
    Walk fake/ and legit/ folders and return list of
    {filename, actual_label, text} dicts.
    """
    files = []

    for label, folder in [("fake", "fake"), ("real", "legit")]:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"  WARNING: folder not found — {folder_path}")
            continue
        for filepath in sorted(glob.glob(os.path.join(folder_path, "*.txt"))):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                files.append({
                    "filename":     os.path.basename(filepath),
                    "actual_label": label,
                    "text":         text,
                })
            except Exception as e:
                print(f"  Could not read {filepath}: {e}")

    print(f"Loaded {len(files)} files "
          f"({sum(1 for f in files if f['actual_label']=='fake')} fake, "
          f"{sum(1 for f in files if f['actual_label']=='real')} real)")
    return files


if __name__ == "__main__":

    all_files = load_all_files(DATASET_PATH)

    # ── Resume from checkpoint ──
    done_files = set()
    results    = []

    if os.path.exists(CHECKPOINT):
        done_df    = pd.read_csv(CHECKPOINT)
        done_files = set(done_df["filename"].tolist())
        results    = done_df.to_dict("records")
        print(f"Resuming — {len(done_files)} already done, "
              f"{len(all_files) - len(done_files)} remaining")
    else:
        print("Starting fresh")

    # ── Run pipeline on each file ──
    for item in tqdm(all_files, desc="Evaluating"):

        if item["filename"] in done_files:
            continue

        try:
            result = evaluate_article(item["text"])
            correct = (result["final_prediction"] == item["actual_label"])

            row = {
                # Identity
                "filename":          item["filename"],
                "actual_label":      item["actual_label"],
                # Final prediction
                "final_prediction":  result["final_prediction"],
                "final_confidence":  result["final_confidence"],
                "correct":           correct,
                # Affective scores
                "vreg":              result["vreg"],
                "eireg_anger":       result["eireg_anger"],
                "eireg_fear":        result["eireg_fear"],
                "eireg_joy":         result["eireg_joy"],
                "eireg_sadness":     result["eireg_sadness"],
                # Logistic regression
                "affect_pred":       result["affect_pred"],
                "affect_confidence": result["affect_confidence"],
                "p_fake":            result["p_fake"],
                # Web verification
                "web_verdict":       result["web_verdict"],
                "web_confidence":    result["web_confidence"],
                "web_explanation":   result["web_explanation"],
                "web_query":         result["web_query"],
                # Fusion internals
                "p_fake_affect":     result["p_fake_affect"],
                "p_fake_web":        result["p_fake_web"],
                "combined_p_fake":   result["combined_p_fake"],
            }

        except Exception as e:
            print(f"  ERROR on {item['filename']}: {e}")
            row = {
                "filename":         item["filename"],
                "actual_label":     item["actual_label"],
                "final_prediction": "ERROR",
                "final_confidence": 0.0,
                "correct":          False,
                "web_verdict":      "ERROR",
                "web_explanation":  str(e),
            }

        results.append(row)
        done_files.add(item["filename"])

        # Checkpoint every 10 files
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
            print(f"  Checkpoint saved — {len(results)} done")
        

    # ── Save full report ──
    report_df = pd.DataFrame(results)
    report_df.to_csv(FINAL_REPORT, index=False)
    print(f"\nSaved → {FINAL_REPORT}")

    # ── Metrics (exclude ERROR rows) ──
    clean = report_df[report_df["final_prediction"] != "ERROR"].copy()
    y_true = (clean["actual_label"]     == "fake").astype(int)
    y_pred = (clean["final_prediction"] == "fake").astype(int)

    print("\n" + "="*55)
    print("EVALUATION RESULTS")
    print("="*55)
    print(f"  Total files:   {len(report_df)}")
    print(f"  Errors:        {(report_df['final_prediction']=='ERROR').sum()}")
    print(f"  Evaluated:     {len(clean)}")
    print(f"\n  Accuracy:      {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1 (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Precision:     {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall:        {recall_score(y_true, y_pred, average='weighted'):.4f}")

    print("\nDetailed report:")
    print(classification_report(y_true, y_pred, target_names=["real","fake"]))

    # ── Breakdown by affect vs web agent ──
    print("="*55)
    print("AGENT BREAKDOWN")
    print("="*55)
    if "affect_pred" in clean.columns:
        y_affect = (clean["affect_pred"] == "fake").astype(int)
        print(f"  Affect-only F1:  {f1_score(y_true, y_affect, average='weighted'):.4f}")

    if "web_verdict" in clean.columns:
        web_clean = clean[clean["web_verdict"].isin(["SUPPORTED","CONTRADICTED"])].copy()
        if len(web_clean) > 0:
            y_true_web = (web_clean["actual_label"] == "fake").astype(int)
            # CONTRADICTED = predicted fake, SUPPORTED = predicted real
            y_web = (web_clean["web_verdict"] == "CONTRADICTED").astype(int)
            print(f"  Web-only F1:     {f1_score(y_true_web, y_web, average='weighted'):.4f}")
            print(f"  Web coverage:    {len(web_clean)}/{len(clean)} articles "
                  f"({100*len(web_clean)/len(clean):.1f}%)")

    print(f"\n  Combined F1:     {f1_score(y_true, y_pred, average='weighted'):.4f}")