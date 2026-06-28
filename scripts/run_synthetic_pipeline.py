from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path("data/synthetic")
OUTPUT_DIR = Path("outputs")
GRAPH_DIR = OUTPUT_DIR / "graphs"
TABLE_DIR = OUTPUT_DIR / "tables"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


USERS_PATH = DATA_DIR / "users.csv"
EDGES_PATH = DATA_DIR / "user_interaction_edges.csv"
POSTS_PATH = DATA_DIR / "posts.csv"


# -----------------------------
# Settings
# -----------------------------

EDGE_THRESHOLD = 0.12

DBSCAN_EPS = 0.80
DBSCAN_MIN_SAMPLES = 3


# -----------------------------
# Load data
# -----------------------------

users = pd.read_csv(USERS_PATH)
edges = pd.read_csv(EDGES_PATH)
posts = pd.read_csv(POSTS_PATH)

print("Loaded data:")
print("Users:", len(users))
print("Edges:", len(edges))
print("Posts:", len(posts))


# -----------------------------
# Build similarity matrix
# -----------------------------

user_ids = sorted(users["user_id"].unique())
user_index = {user_id: i for i, user_id in enumerate(user_ids)}

n_users = len(user_ids)

similarity_matrix = np.zeros((n_users, n_users))
np.fill_diagonal(similarity_matrix, 1.0)

for _, row in edges.iterrows():
    user_1 = row["user_1"]
    user_2 = row["user_2"]

    if user_1 not in user_index or user_2 not in user_index:
        continue

    i = user_index[user_1]
    j = user_index[user_2]

    similarity = float(row["final_interaction_weight"])

    similarity_matrix[i, j] = similarity
    similarity_matrix[j, i] = similarity


# -----------------------------
# Convert similarity to distance
# -----------------------------

distance_matrix = 1 - similarity_matrix
np.fill_diagonal(distance_matrix, 0.0)


# -----------------------------
# Cluster users
# -----------------------------

model = DBSCAN(
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES,
    metric="precomputed"
)

cluster_labels = model.fit_predict(distance_matrix)

clusters = pd.DataFrame({
    "user_id": user_ids,
    "detected_cluster": cluster_labels
})

clusters = clusters.merge(
    users[[
        "user_id",
        "username",
        "ground_truth_cluster",
        "ground_truth_stance",
        "cluster_label"
    ]],
    on="user_id",
    how="left"
)

clusters.to_csv(TABLE_DIR / "detected_user_clusters.csv", index=False)

print()
print("Detected clusters:")
print(clusters["detected_cluster"].value_counts().sort_index())


# -----------------------------
# Evaluate against ground truth
# -----------------------------

# Convert ground-truth labels to numeric codes
ground_truth_codes = clusters["ground_truth_cluster"].astype("category").cat.codes

ari = adjusted_rand_score(
    ground_truth_codes,
    clusters["detected_cluster"]
)

print()
print(f"Adjusted Rand Index against synthetic ground truth: {ari:.3f}")


# -----------------------------
# Build graph
# -----------------------------

G = nx.Graph()

cluster_lookup = clusters.set_index("user_id")["detected_cluster"].to_dict()
username_lookup = clusters.set_index("user_id")["username"].to_dict()
truth_lookup = clusters.set_index("user_id")["ground_truth_cluster"].to_dict()

for user_id in user_ids:
    G.add_node(
        user_id,
        username=username_lookup.get(user_id, user_id),
        detected_cluster=int(cluster_lookup.get(user_id, -1)),
        ground_truth_cluster=truth_lookup.get(user_id, "")
    )

filtered_edges = edges[
    edges["final_interaction_weight"] >= EDGE_THRESHOLD
].copy()

for _, row in filtered_edges.iterrows():
    G.add_edge(
        row["user_1"],
        row["user_2"],
        weight=float(row["final_interaction_weight"])
    )

print()
print("Graph:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# -----------------------------
# Visualize graph
# -----------------------------

pos = nx.spring_layout(
    G,
    weight="weight",
    seed=42,
    k=0.35
)

node_colors = [
    G.nodes[node]["detected_cluster"]
    for node in G.nodes()
]

edge_widths = [
    G[u][v]["weight"] * 8
    for u, v in G.edges()
]

plt.figure(figsize=(16, 12))

nx.draw_networkx_edges(
    G,
    pos,
    width=edge_widths,
    alpha=0.20
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    cmap=plt.cm.Set3,
    node_size=90,
    edgecolors="black",
    linewidths=0.25
)

plt.title(
    "Synthetic Echo Chamber Graph: Detected User Clusters",
    fontsize=16
)

plt.axis("off")
plt.tight_layout()

graph_output = GRAPH_DIR / "synthetic_echo_chamber_graph.png"
plt.savefig(graph_output, dpi=300)
plt.show()

print()
print(f"Graph saved to: {graph_output}")


# -----------------------------
# Cluster summary
# -----------------------------

cluster_summary = (
    clusters
    .groupby("detected_cluster")
    .agg(
        user_count=("user_id", "count"),
        dominant_ground_truth_cluster=(
            "ground_truth_cluster",
            lambda values: values.value_counts().index[0]
        ),
        dominant_stance=(
            "ground_truth_stance",
            lambda values: values.value_counts().index[0]
        )
    )
    .reset_index()
)

cluster_summary.to_csv(TABLE_DIR / "cluster_summary.csv", index=False)

print()
print("Cluster summary:")
print(cluster_summary)