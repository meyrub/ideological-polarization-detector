from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

USERS_CSV = DATA_DIR / "users_project_ready.csv"
POSTS_CSV = DATA_DIR / "posts_cleaned_project_ready.csv"
EDGES_CSV = DATA_DIR / "inferred_user_edges_graph_ready.csv"
HASHTAGS_CSV = DATA_DIR / "post_hashtags_project_ready.csv"

GRAPH_OUTPUT = OUTPUT_DIR / "graphs" / "echo_chamber_graph.png"
CLUSTER_OUTPUT = OUTPUT_DIR / "tables" / "user_clusters.csv"
POLARIZATION_OUTPUT = OUTPUT_DIR / "tables" / "polarization_report.csv"

# Graph settings
SIMILARITY_THRESHOLD = 0.12

# DBSCAN settings
DBSCAN_EPS = 0.80
DBSCAN_MIN_SAMPLES = 3

# Similarity weights
SIMILARITY_WEIGHTS = {
    "semantic_similarity": 0.40,
    "hashtag_jaccard": 0.25,
    "sentiment_family_jaccard": 0.15,
    "same_primary_country": 0.10,
    "same_primary_platform": 0.05,
    "engagement_similarity": 0.05,
}

# Polarization weights
POLARIZATION_WEIGHTS = {
    "structural": 0.40,
    "modularity": 0.30,
    "semantic": 0.30,
}