from echo_chamber.config import (
    USERS_CSV,
    POSTS_CSV,
    EDGES_CSV,
    HASHTAGS_CSV,
    GRAPH_OUTPUT,
    CLUSTER_OUTPUT,
    POLARIZATION_OUTPUT,
    SIMILARITY_THRESHOLD,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
)

from echo_chamber.data_loader import load_project_csvs
from echo_chamber.similarity import build_similarity_matrix, build_distance_matrix
from echo_chamber.clustering import cluster_users_dbscan
from echo_chamber.graph_builder import build_graph, visualize_graph
from echo_chamber.polarization import build_polarization_report
from echo_chamber.export import export_project_results


def main():
    data = load_project_csvs(
        users_path=USERS_CSV,
        posts_path=POSTS_CSV,
        edges_path=EDGES_CSV,
        hashtags_path=HASHTAGS_CSV
    )

    users = data["users"]
    posts = data["posts"]
    edges = data["edges"]

    user_ids, similarity_matrix = build_similarity_matrix(
        users=users,
        edges=edges
    )

    distance_matrix = build_distance_matrix(similarity_matrix)

    clusters = cluster_users_dbscan(
        user_ids=user_ids,
        distance_matrix=distance_matrix,
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES
    )

    graph = build_graph(
        users=users,
        clusters=clusters,
        edges=edges,
        threshold=SIMILARITY_THRESHOLD
    )

    visualize_graph(
        graph=graph,
        output_path=GRAPH_OUTPUT,
        title="Detected Echo Chambers Based on Inferred User Similarity"
    )

    polarization_report = build_polarization_report(
        graph=graph,
        posts=posts,
        clusters=clusters
    )

    export_project_results(
        clusters=clusters,
        polarization_report=polarization_report,
        cluster_output=CLUSTER_OUTPUT,
        polarization_output=POLARIZATION_OUTPUT
    )

    print("Pipeline complete.")
    print(f"Graph saved to: {GRAPH_OUTPUT}")
    print(f"Clusters saved to: {CLUSTER_OUTPUT}")
    print(f"Polarization report saved to: {POLARIZATION_OUTPUT}")


if __name__ == "__main__":
    main()