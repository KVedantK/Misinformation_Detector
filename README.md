# Cross-Domain Misinformation Detection

MSc Project — Extended RAEmoLLM framework combining affective embeddings, emotional retrieval, source credibility, and live web verification.

---

## Overview

```
FakeNewsNet CSVs
(BuzzFeed + PolitiFact)
        │
        ▼
  Data Cleaning
  + full_text col
        │
        ▼
llama_cpp_for_emollama_gguf.py
  ├── Vreg / EIreg / Voc / Ec scores
  └── Affective embeddings (422 × 4096)
        │
        ▼
web_verification.py
  ├── Chain 1: LLM generates query → Tavily searches
  └── Chain 2: LLM reads evidence → verdict JSON
        │
        ▼
RAEmoLLM Pipeline
(Retrieval + Inference)
```

---

## Prerequisites

```bash
conda create -n msc_project python=3.11
conda activate msc_project

pip install pandas numpy scikit-learn scipy tqdm matplotlib seaborn huggingface_hub
pip install langchain-huggingface langchain-community tavily-python

# llama-cpp with Metal (Apple Silicon)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## API Keys

| Service | Where to get |
|---------|-------------|
| HuggingFace token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Tavily key | [app.tavily.com](https://app.tavily.com) — 1000 free searches/month |

---

## Step 1 — Download EmoLLaMA GGUF

```bash
huggingface-cli login   # paste your HF token when prompted

mkdir -p emollama_gguf

huggingface-cli download mradermacher/Emollama-7b-GGUF \
  Emollama-7b.Q4_K_M.gguf \
  --local-dir ./emollama_gguf
```

The model file is ~4.5GB and uses ~7-8GB RAM at runtime — comfortable on M4 16GB.

---

## Step 2 — Data Cleaning

- Load the four FakeNewsNet CSVs, drop unused columns, create the `full_text` field:
- We are to drop columns like id, image, author, publish dates, cannonical images / links, meta data that are not important for the analysis.
- We are also combining the tittle and the content of the article to obtain the full text column that is useful for the LLM analysis. 

---

## Step 3 — Affective Scoring + Embeddings

Run `llama_cpp_for_emollama_gguf.py`. This does two things in sequence:

**Part A — Explicit scores** using EmoLLaMA generation mode:

Extracts `vreg`, `voc`, `eireg_anger`, `eireg_fear`, `eireg_joy`, `eireg_sadness`, `ec` per article using the exact EmoLLMs prompt templates:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./emollama_gguf/Emollama-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)
```

Prompts used:

```python
# Vreg — sentiment strength 0 (negative) → 1 (positive)
"Human:\nTask: Evaluate the valence intensity of the writer's mental state "
"based on the text, assigning it a real-valued score from 0 (most negative) "
"to 1 (most positive).\nText: {text}\nIntensity Score:\n\nA:\n>>"

# EIreg — emotion intensity per emotion (anger / fear / joy / sadness)
"Human:\nTask: Assign a numerical value between 0 (least E) and 1 (most E) "
"to represent the intensity of emotion E expressed in the text.\n"
"Text: {text}\nEmotion: {emotion}\nIntensity Score:\n\nA:\n>>"

# Voc — sentiment class -3 to 3
"Human:\nTask: Categorize the text into an ordinal class... "
"3: very positive ... -3: very negative\n"
"Text: {text}\nIntensity Class:\n\nA:\n>>"

# Ec — which emotions are present
"Human:\nTask: Categorize the text's emotional tone as either "
"'neutral or no emotion' or identify emotions "
"(anger, anticipation, disgust, fear, joy, love, optimism, pessimism, "
"sadness, surprise, trust).\nText: {text}\nThis text contains emotions:\n\nA:\n>>"
```

Saves checkpoint every 25 rows → `profiles_checkpoint.csv`. Resumes automatically if interrupted.

**Part B — Implicit embeddings** using EmoLLaMA embedding mode:

```python
llm_embed = Llama(
    model_path="./emollama_gguf/Emollama-7b.Q4_K_M.gguf",
    embedding=True,
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)
```

Extracts 4096-dimensional last hidden layer vector per article (under Vreg prompt context) and writes directly to disk one row at a time — no full array ever loaded into RAM:

```python
fp = np.lib.format.open_memmap("affective_embeddings.npy",
                                mode="r+", dtype=np.float32,
                                shape=(total, 4096))
fp[i] = emb
fp.flush()
del fp
```

**Outputs after Step 3:**

```
fakenewsnet_with_affect.csv      — 422 rows with all affect scores
affective_embeddings.npy         — shape (422, 4096) float32
```

> **Note:** Load generation mode and embedding mode separately — not simultaneously — to stay within 16GB RAM.

---

## Step 4 — Web Verification

Run `web_verification.py`. Two separate LLM chains:

```python
os.environ["HF_TOKEN"]       = "hf_your_token"
os.environ["TAVILY_API_KEY"] = "tvly_your_key"
```

**Chain 1 — Evidence retrieval:**

LLM generates a focused search query from the article title → Tavily fetches top 5 results → returns snippets and source URLs. No verdict here — retrieval only.

**Chain 2 — Verdict generation:**

LLM reads the evidence snippets and outputs structured JSON:

```json
{
  "verdict": "SUPPORTED" or "CONTRADICTED",
  "confidence": 0.85,
  "explanation": "Three sources confirm the claim...",
  "supporting_sources": ["https://..."],
  "contradicting_sources": []
}
```

Checkpoints every 10 articles → `web_verification_checkpoint.csv`. Resumes automatically.

**Output after Step 4:**

```
web_verification_results.csv     — verdict + confidence + explanation per article
```

---

## Step 5 — RAEmoLLM Pipeline

With `fakenewsnet_with_affect.csv`, `affective_embeddings.npy`, and `web_verification_results.csv` all ready, run `raemollm_pipeline.py`.

For each target article:
1. Cosine similarity in emotional embedding space → top-4 retrieved examples from source domain
2. Build Template 2 prompt with retrieved examples + Vreg scores + web verdict
3. Mistral-7B classifies: **Fake** or **Legit**
4. Evaluate with Accuracy, Precision, Recall, F1 (weighted)

Results saved to `raemollm_results.csv`.
