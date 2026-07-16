import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist

df = pd.read_csv("fakenewsnet_with_affect.csv")

FEATURES = ["vreg", "eireg_anger", "eireg_fear", "eireg_joy", "eireg_sadness"]

X = df[FEATURES].values
y = (df["label"] == "fake").astype(int).values   # 1=fake, 0=real

# Always scale — KMeans is distance-based
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# APPROACH 1 — KMeans (k=2)
# Simplest — finds two cluster centres
# ─────────────────────────────────────────
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
kmeans.fit(X_scaled)

# Map cluster IDs to fake/real labels
# (KMeans doesn't know which cluster is fake)
cluster_labels = kmeans.labels_

# Figure out which cluster = fake by majority vote
cluster_0_fake = (y[cluster_labels == 0] == 1).mean()
cluster_1_fake = (y[cluster_labels == 1] == 1).mean()

if cluster_0_fake > cluster_1_fake:
    pred_kmeans = cluster_labels          # cluster 0 = fake
else:
    pred_kmeans = 1 - cluster_labels      # cluster 1 = fake

print("=== KMeans (k=2) ===")
print(classification_report(y, pred_kmeans, target_names=["real","fake"]))

# ─────────────────────────────────────────
# APPROACH 2 — Nearest Centroid
# Compute mean affective profile per class
# New item → distance to each centroid → closest wins
# This is the most interpretable and closest
# to what RAEmoLLM does conceptually
# ─────────────────────────────────────────
from sklearn.neighbors import NearestCentroid

nc = NearestCentroid()
nc.fit(X_scaled, y)

# Show what the centroids look like
centroid_df = pd.DataFrame(
    scaler.inverse_transform(nc.centroids_),
    columns=FEATURES,
    index=["real centroid", "fake centroid"]
)
print("\n=== Cluster Centroids (original scale) ===")
print(centroid_df.round(4))

pred_nc = nc.predict(X_scaled)
print("\n=== Nearest Centroid ===")
print(classification_report(y, pred_nc, target_names=["real","fake"]))

# Cross-validated F1
cv_f1 = cross_val_score(nc, X_scaled, y, cv=5, scoring="f1_weighted")
print(f"5-fold CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# ─────────────────────────────────────────
# APPROACH 3 — Gaussian Mixture Model
# Soft clustering — models probability
# distributions around each class centre
# Better when classes overlap (which yours do)
# ─────────────────────────────────────────
gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
gmm.fit(X_scaled)

gmm_labels    = gmm.predict(X_scaled)
gmm_probs     = gmm.predict_proba(X_scaled)

# Map components to labels
comp_0_fake = (y[gmm_labels == 0] == 1).mean()
comp_1_fake = (y[gmm_labels == 1] == 1).mean()
fake_component = 0 if comp_0_fake > comp_1_fake else 1

pred_gmm = (gmm_labels == fake_component).astype(int)
print("\n=== Gaussian Mixture Model ===")
print(classification_report(y, pred_gmm, target_names=["real","fake"]))

# ─────────────────────────────────────────
# APPROACH 4 — Affective Distance Score
# Most useful for YOUR pipeline specifically
# For a new article: compute distance to
# fake centroid vs real centroid
# This gives a continuous score you can
# feed INTO your RAEmoLLM Template 2 prompt
# ─────────────────────────────────────────
fake_centroid = X_scaled[y == 1].mean(axis=0)
real_centroid = X_scaled[y == 0].mean(axis=0)

def affective_distance_score(scores: dict) -> dict:
    """
    Given affect scores for a new article,
    compute how close it is to fake vs real centroid.
    Returns a score and predicted label.
    """
    x = np.array([[scores[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)

    dist_to_fake = np.linalg.norm(x_scaled - fake_centroid)
    dist_to_real = np.linalg.norm(x_scaled - real_centroid)

    # Normalised score: 0=clearly real, 1=clearly fake
    affective_score = dist_to_real / (dist_to_fake + dist_to_real)

    return {
        "affective_score":  round(affective_score, 4),
        "dist_to_fake":     round(dist_to_fake, 4),
        "dist_to_real":     round(dist_to_real, 4),
        "cluster_pred":     "fake" if dist_to_fake < dist_to_real else "real"
    }

# Test on your dataset
df["affective_score"] = 0.0
df["cluster_pred"]    = ""

for i, row in df.iterrows():
    scores = {f: row[f] for f in FEATURES}
    result = affective_distance_score(scores)
    df.at[i, "affective_score"] = result["affective_score"]
    df.at[i, "cluster_pred"]    = result["cluster_pred"]

pred_dist = (df["cluster_pred"] == "fake").astype(int)
print("\n=== Affective Distance Score ===")
print(classification_report(y, pred_dist, target_names=["real","fake"]))

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*50)
print("SUMMARY — F1 Weighted")
print("="*50)
for name, pred in [
    ("KMeans",            pred_kmeans),
    ("Nearest Centroid",  pred_nc),
    ("GMM",               pred_gmm),
    ("Affective Distance",pred_dist),
]:
    f1  = f1_score(y, pred, average="weighted")
    acc = accuracy_score(y, pred)
    print(f"{name:<25} F1={f1:.4f}  Acc={acc:.4f}")

# Save dataset with cluster predictions
df.to_csv("fakenewsnet_with_clusters.csv", index=False)
print("\nSaved → fakenewsnet_with_clusters.csv")