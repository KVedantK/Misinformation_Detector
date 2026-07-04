from llama_cpp import Llama
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# ─────────────────────────────────────────
# LOAD EMBEDDING MODEL ONLY
# ─────────────────────────────────────────
llm_embed = Llama(
    model_path="./emollama_gguf/Emollama-7b.Q4_K_M.gguf",
    embedding=True,
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

# ─────────────────────────────────────────
# VREG PROMPT — embeddings extracted 
# mid-prompt as per paper
# ─────────────────────────────────────────
def prompt_vreg(text: str) -> str:
    return (
        f"Human:\nTask: Evaluate the valence intensity of the writer's mental state "
        f"based on the text, assigning it a real-valued score from 0 (most negative) "
        f"to 1 (most positive).\nText: {text}\nIntensity Score:\n\nA:\n>>"
    )

# ─────────────────────────────────────────
# GET SINGLE EMBEDDING — no list stored
# ─────────────────────────────────────────
def get_embedding(text: str) -> np.ndarray:
    """Extract embedding using Vreg prompt context (as per paper)"""
    for max_chars in [250, 150, 80]:
        try:
            emb = llm_embed.embed(prompt_vreg(text[:max_chars]))
            emb = np.array(emb, dtype=np.float32)
            
            # handle token-level output — mean pool to single vector
            if emb.ndim == 2:
                emb = emb.mean(axis=0)   # (n_tokens, 4096) → (4096,)
            elif emb.ndim == 3:
                emb = emb.mean(axis=(0, 1))  # edge case
                
            return emb  # shape (4096,)
            
        except RuntimeError as e:
            if "llama_decode returned -3" in str(e):
                print(f"  Retrying with {max_chars} chars...")
                continue
            raise e
            
    print("  WARNING: embedding failed, returning zeros")
    return np.zeros(4096, dtype=np.float32)

# ─────────────────────────────────────────
# APPEND ONE ROW TO .NPY FILE AT A TIME
# Uses memory-mapped file — never loads
# the full array into RAM
# ─────────────────────────────────────────

EMBED_FILE  = "affective_embeddings.npy"
PROGRESS_FILE = "embed_progress.txt"

df = pd.read_csv("ready_data.csv")
total = len(df)

# ── Check progress ──
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        start_idx = int(f.read().strip())
    print(f"Resuming from row {start_idx} / {total}")
else:
    start_idx = 0
    print("Starting fresh")

# ── Initialise .npy file if starting fresh ──
if start_idx == 0:
    # Create empty array on disk — shape (total, 4096)
    # This pre-allocates the full file without loading it into RAM
    fp = np.lib.format.open_memmap(
        EMBED_FILE,
        mode="w+",               # write + read
        dtype=np.float32,
        shape=(total, 4096)
    )
    del fp                       # close immediately — don't hold in RAM
    print(f"Initialised {EMBED_FILE} with shape ({total}, 4096)")
else:
    print(f"Appending to existing {EMBED_FILE}")

# ─────────────────────────────────────────
# PROCESS ONE BY ONE — write directly to disk
# ─────────────────────────────────────────

for i, row in tqdm(df.iloc[start_idx:].iterrows(),
                   total=total - start_idx,
                   desc="Extracting embeddings"):

    text = row["full_text"]

    # get single embedding — not stored in any list
    emb = get_embedding(text)

    # open memmap, write one row, close immediately
    fp = np.lib.format.open_memmap(
        EMBED_FILE,
        mode="r+",               # read + write existing file
        dtype=np.float32,
        shape=(total, 4096)
    )
    fp[i] = emb                  # write just this one row
    fp.flush()                   # force write to disk
    del fp                       # release from RAM immediately

    # save progress index
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(i + 1))

    if (i + 1) % 25 == 0:
        print(f"  ✓ {i+1}/{total} embeddings saved to disk")

# ─────────────────────────────────────────
# VERIFY FINAL FILE
# ─────────────────────────────────────────
fp_check = np.load(EMBED_FILE, mmap_mode="r")   # read-only memmap — no RAM usage
print(f"\nDone. Embedding file shape: {fp_check.shape}")
print(f"Sample row 0 norm: {np.linalg.norm(fp_check[0]):.4f}")
print(f"Sample row 1 norm: {np.linalg.norm(fp_check[1]):.4f}")
del fp_check