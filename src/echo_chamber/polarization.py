import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from networkx.algorithms.community.quality import modularity


def structural_polarization_score(graph):
    """
    Calculate structural polarization.

    Formula:
        (internal_edge_weight - external_edge_weight)
        /
        (internal_edge_weight + external_edge_weight)
    """

    internal_weight = 0.0
    external_weight = 0.0

    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)

        cluster_u = graph.nodes[u].get("cluster", -1)
        cluster_v = graph.nodes[v].get("cluster", -1)

        if cluster_u == -1 or cluster_v == -1:
            continue

        if cluster_u == cluster_v:
            internal_weight += weight
        else:
            external_weight += weight

    total = internal_weight + external_weight

    if total == 0:
        score = 0.0
    else:
        score = (internal_weight - external_weight) / total

    return {
        "structural_polarization": score,
        "internal_edge_weight": internal_weight,
        "external_edge_weight": external_weight
    }


def calculate_modularity(graph):
    """
    Calculate graph modularity based on detected clusters.
    """

    clusters = {}

    for node, data in graph.nodes(data=True):
        cluster = data.get("cluster", -1)

        if cluster == -1:
            continue

        clusters.setdefault(cluster, set()).add(node)

    communities = list(clusters.values())

    if len(communities) < 2:
        return 0.0

    return modularity(graph, communities, weight="weight")


def cross_cluster_exposure(graph):
    """
    Calculate cross-cluster exposure.

    Formula:
        external_edge_weight / total_edge_weight
    """

    result = structural_polarization_score(graph)

    internal = result["internal_edge_weight"]
    external = result["external_edge_weight"]

    total = internal + external

    if total == 0:
        return 0.0

    return external / total


def semantic_polarization(posts, clusters):
    """
    Measure semantic distance between clusters.

    Uses post text grouped by detected cluster.
    """

    if "text" not in posts.columns:
        raise ValueError("posts must contain a 'text' column")

    merged = posts.merge(
        clusters[["user_id", "cluster"]],
        on="user_id",
        how="inner"
    )

    merged = merged[merged["cluster"] != -1]

    if merged["cluster"].nunique() < 2:
        return 0.0

    cluster_documents = (
        merged
        .groupby("cluster")["text"]
        .apply(lambda texts: " ".join(texts.dropna().astype(str)))
        .reset_index()
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=1
    )

    matrix = vectorizer.fit_transform(cluster_documents["text"])

    distance_matrix = cosine_distances(matrix)

    upper_triangle = distance_matrix[
        np.triu_indices_from(distance_matrix, k=1)
    ]

    return float(upper_triangle.mean())


def normalize_to_0_1(value, min_value=-1, max_value=1):
    """
    Normalize a value from one range to the 0-1 range.
    """

    return (value - min_value) / (max_value - min_value)


def final_polarization_index(
    structural_score,
    modularity_score,
    semantic_score,
    weights=None
):
    """
    Combine polarization metrics into one final index.
    """

    if weights is None:
        weights = {
            "structural": 0.40,
            "modularity": 0.30,
            "semantic": 0.30,
        }

    structural_norm = normalize_to_0_1(
        structural_score,
        min_value=-1,
        max_value=1
    )

    index = (
        weights["structural"] * structural_norm
        + weights["modularity"] * modularity_score
        + weights["semantic"] * semantic_score
    )

    return index


def build_polarization_report(graph, posts, clusters):
    """
    Build a presentation-ready polarization report.
    """

    structural = structural_polarization_score(graph)

    modularity_score = calculate_modularity(graph)
    semantic_score = semantic_polarization(posts, clusters)
    exposure_score = cross_cluster_exposure(graph)

    polarization_index = final_polarization_index(
        structural["structural_polarization"],
        modularity_score,
        semantic_score
    )

    report = pd.DataFrame({
        "metric": [
            "Structural polarization",
            "Internal edge weight",
            "External edge weight",
            "Modularity",
            "Semantic polarization",
            "Cross-cluster exposure",
            "Final polarization index"
        ],
        "score": [
            structural["structural_polarization"],
            structural["internal_edge_weight"],
            structural["external_edge_weight"],
            modularity_score,
            semantic_score,
            exposure_score,
            polarization_index
        ],
        "interpretation": [
            "Higher means more interaction stays inside clusters",
            "Total strength of within-cluster edges",
            "Total strength of between-cluster edges",
            "Higher means stronger community separation",
            "Higher means clusters are more textually different",
            "Lower means less cross-cluster interaction",
            "Combined estimate of overall polarization"
        ]
    })

    return report