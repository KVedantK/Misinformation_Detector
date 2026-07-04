# EmoLLaMA GGUF Setup Guide

Getting EmoLLaMA-7B running locally in GGUF format for affective embedding extraction and sentiment scoring.

---

## Prerequisites

Make sure you have the following before starting:

- Python 3.10+
- Conda environment (recommended)
- Apple Silicon Mac (M1/M2/M3/M4) **or** a machine with CUDA GPU
- ~5GB free disk space for the model file
- A HuggingFace account (free) for downloading the model

---

## Step 1 — Install Dependencies

Activate your conda environment first:

```bash
conda activate msc_project
```

Install `llama-cpp-python` with Metal acceleration (Apple Silicon):

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

If you are on a CUDA machine instead:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Install the HuggingFace CLI for downloading:

```bash
pip install huggingface_hub
```

---

## Step 2 — Log In to HuggingFace

You need a free HuggingFace account to download the model.

```bash
huggingface-cli login
```

Paste your token when prompted. You can create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — a read-only token is sufficient.

---

## Step 3 — Create the Model Folder

Navigate to your project root (wherever your Python scripts live) and create the folder:

```bash
cd /path/to/your/project
mkdir -p emollama_gguf
```

Your project structure should look like:

```
your_project/
├── emollama_gguf/          ← model goes here
├── fakenewsnet_with_affect.csv
├── affective_embeddings.npy
├── embeddings_saver.py
└── ...
```

---

## Step 4 — Download the GGUF Model

Run this from your **project root** directory:

```bash
huggingface-cli download mradermacher/Emollama-7b-GGUF \
  Emollama-7b.Q4_K_M.gguf \
  --local-dir ./emollama_gguf
```

This downloads the **Q4_K_M quantized** version (~4.5GB). This is the recommended variant — best balance of quality and memory usage for a 16GB machine.

To verify the download completed successfully:

```bash
ls -lh ./emollama_gguf/
```

You should see:

```
-rw-r--r--  1 user  staff   4.5G  Emollama-7b.Q4_K_M.gguf
```

---

## Step 5 — Verify the Model Loads

Run this quick sanity check before running the full pipeline:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./emollama_gguf/Emollama-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # -1 = use all GPU layers (Metal on Mac, CUDA on Linux)
    verbose=False
)

# Quick generation test
out = llm(
    "Human:\nTask: Evaluate the valence intensity of the writer's mental state "
    "based on the text, assigning it a real-valued score from 0 (most negative) "
    "to 1 (most positive).\nText: I love this!\nIntensity Score:\n\nA:\n>>",
    max_tokens=10,
    temperature=0,
    echo=False
)
print("Generation test:", out["choices"][0]["text"].strip())

# Quick embedding test
emb = llm.embed("This is a test sentence.")
import numpy as np
emb_array = np.array(emb)
print("Embedding shape:", emb_array.shape)
print("Embedding test passed ✓")
```

Expected output:

```
Generation test: 0.9
Embedding shape: (4096,)  or  (N, 4096)
Embedding test passed ✓
```

---

## Step 6 — Use in Your Scripts

The model path used across all project scripts is:

```python
from llama_cpp import Llama

# Generation mode — for explicit affect scores (Vreg, EIreg, Voc, Ec)
llm = Llama(
    model_path="./emollama_gguf/Emollama-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

# Embedding mode — for implicit affective embeddings (retrieval database)
llm_embed = Llama(
    model_path="./emollama_gguf/Emollama-7b.Q4_K_M.gguf",
    embedding=True,   # switches to embedding mode
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)
```

> **Important:** Do not load both instances at the same time — load one, finish all operations, then load the other. On a 16GB machine running both simultaneously will exhaust memory.

---

## Memory Usage Reference

| Variant | File Size | RAM Used | Recommended For |
|---------|-----------|----------|-----------------|
| Q4_K_M  | ~4.5 GB   | ~7-8 GB  | M4 16GB — this one |
| Q5_K_M  | ~5.3 GB   | ~8-9 GB  | M4 16GB (higher quality) |
| Q8_0    | ~7.2 GB   | ~10 GB   | M4 16GB (near full quality) |
| F16     | ~13.5 GB  | ~15 GB   | Too tight for 16GB |

---

## Troubleshooting

**`RuntimeError: llama_decode returned -3`**

The input text is too long for the context window. Truncate your text before passing it:

```python
text = text[:250]  # reduce until error goes away
```

**`Metal backend not found`**

Reinstall with Metal explicitly enabled:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**`Embedding shape is (N, 4096) not (4096,)`**

The model returns one vector per token. Mean-pool to get a single sentence vector:

```python
import numpy as np
emb = np.array(llm_embed.embed(text))
if emb.ndim == 2:
    emb = emb.mean(axis=0)  # (N, 4096) → (4096,)
```

**Download fails halfway**

Re-run the same `huggingface-cli download` command — it resumes from where it stopped automatically.

**`ModuleNotFoundError: llama_cpp`**

The build failed silently. Check your Xcode command line tools are installed:

```bash
xcode-select --install
```

Then reinstall llama-cpp-python.

---

## File Structure After Setup

```
your_project/
├── emollama_gguf/
│   └── Emollama-7b.Q4_K_M.gguf     ← 4.5GB model file
├── affective_embeddings.npy          ← (422, 4096) embeddings
├── profiles_checkpoint.csv           ← explicit affect scores
├── fakenewsnet_with_affect.csv       ← full dataset with scores
├── embeddings_saver.py               ← embedding extraction script
└── web_verification_results.csv      ← Tavily verification output
```

---

## Quick Reference — All EmoLLaMA Prompts

These are the exact prompt formats EmoLLaMA was trained on:

```python
# Vreg — sentiment strength (0=negative, 1=positive)
f"Human:\nTask: Evaluate the valence intensity of the writer's mental state "
f"based on the text, assigning it a real-valued score from 0 (most negative) "
f"to 1 (most positive).\nText: {text}\nIntensity Score:\n\nA:\n>>"

# EIreg — emotion intensity for anger/fear/joy/sadness
f"Human:\nTask: Assign a numerical value between 0 (least E) and 1 (most E) "
f"to represent the intensity of emotion E expressed in the text.\n"
f"Text: {text}\nEmotion: {emotion}\nIntensity Score:\n\nA:\n>>"

# Voc — sentiment classification (-3 to 3)
f"Human:\nTask: Categorize the text into an ordinal class that best characterizes "
f"the writer's mental state. 3: very positive. 2: moderately positive. "
f"1: slightly positive. 0: neutral. -1: slightly negative. "
f"-2: moderately negative. -3: very negative.\n"
f"Text: {text}\nIntensity Class:\n\nA:\n>>"

# Ec — emotion detection
f"Human:\nTask: Categorize the text's emotional tone as either "
f"'neutral or no emotion' or identify the presence of one or more of the given "
f"emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, "
f"sadness, surprise, trust).\nText: {text}\nThis text contains emotions:\n\nA:\n>>"
```
