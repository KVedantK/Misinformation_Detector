"""
Generates & saves implicit EmoLLaMA embeddings for fakenewsnet_combined_cleaned.csv.
Does NOT redo vreg/eireg (already in ready_data.csv) — embeddings only.
Saves incrementally to embeddings_checkpoint.npy + row index csv, resumable.
"""
from llama_cpp import Llama
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

MODEL_PATH = "../MSc_Project/emollama_gguf/Emollama-7b.Q4_K_M.gguf"
SRC_CSV = "./Data_Folder_CSVs/fakenewsnet_combined_cleaned.csv"
OUT_NPY = "./Data_Folder_CSVs/fakenewsnet_embeddings.npy"
OUT_IDS = "./Data_Folder_CSVs/fakenewsnet_embeddings_ids.csv"
TEXT_COL = "full_text"
MAX_CHARS = 1000  # truncate like the profile script did

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=-1,
    embedding=True,   # <-- key flag for embedding mode
    verbose=False,
)

df = pd.read_csv(SRC_CSV)

# resume support
start_idx = 0
embeddings = []
if os.path.exists(OUT_NPY) and os.path.exists(OUT_IDS):
    embeddings = list(np.load(OUT_NPY))
    done_ids = pd.read_csv(OUT_IDS)
    start_idx = len(done_ids)
    print(f"Resuming from row {start_idx}")

for i in tqdm(range(start_idx, len(df))):
    text = str(df.iloc[i][TEXT_COL])[:MAX_CHARS]
    try:
        out = llm.embed(text)
        vec = np.array(out, dtype=np.float32)
        if vec.ndim > 1:          # some builds return token-level; mean-pool
            vec = vec.mean(axis=0)
    except Exception as e:
        print(f"row {i} failed: {e}")
        vec = np.zeros(4096, dtype=np.float32)  # placeholder, adjust to model dim

    embeddings.append(vec)

    if (i + 1) % 25 == 0 or i == len(df) - 1:
        np.save(OUT_NPY, np.stack(embeddings))
        pd.DataFrame({"row_idx": range(len(embeddings))}).to_csv(OUT_IDS, index=False)

np.save(OUT_NPY, np.stack(embeddings))
pd.DataFrame({"row_idx": range(len(embeddings))}).to_csv(OUT_IDS, index=False)
print(f"Done. Saved {len(embeddings)} embeddings, dim={embeddings[0].shape[0]} -> {OUT_NPY}")