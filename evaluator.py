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

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
EMOLLAMA_PATH = "./emollama_gguf/Emollama-7b.Q4_K_M.gguf"
MODEL_PATH    = "models/affect_classifier.pkl"
FEATURES      = ["vreg", "eireg_anger", "eireg_fear", "eireg_joy", "eireg_sadness"]

os.environ["HF_TOKEN"]                 = os.getenv("HF_TOKEN")
os.environ["TAVILY_API_KEY"]           = os.getenv("TAVILY_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]

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

llm_web = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        provider="auto",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        max_new_tokens=800,
        temperature=0.01,
        timeout=120,
    ),
    verbose=False
)
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
if __name__ == "__main__":

    sample = """
    Trump Just Made A Campaign Promise So Ridiculous It Makes Read My Lips Look Good.
    Trump just promised to destroy the existential threat of terrorism in America.
    He promised to destroy all terrorists and end terrorism on our soil without any
    specific plan for counter-terrorism funding or internet recruitment prevention.
    """

    result = evaluate_article(sample)

    print("\n" + "="*60)
    print("FULL RESULT SUMMARY")
    print("="*60)
    print(f"Affective scores:    vreg={result['vreg']:.3f}  anger={result['eireg_anger']:.3f}  fear={result['eireg_fear']:.3f}")
    print(f"Affect prediction:   {result['affect_pred']} ({result['affect_confidence']:.4f})")
    print(f"Web verdict:         {result['web_verdict']} ({result['web_confidence']:.2f})")
    print(f"Web explanation:     {result['web_explanation']}")
    print(f"Final prediction:    {result['final_prediction'].upper()} ({result['final_confidence']:.4f})")