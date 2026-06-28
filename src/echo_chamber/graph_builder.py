import matplotlib.pyplot as plt
import networkx as nx


def build_graph(users, clusters, edges, threshold=0.35):
    """
    Build a NetworkX graph from users, clusters, and edges.

    Nodes = users
    Edges = inferred similarity relationships
    Node attribute = cluster
    Edge attribute = final_similarity
    """

    graph = nx.Graph()

    cluster_lookup = clusters.set_index("user_id")["cluster"].to_dict()

    for _, row in users.iterrows():
        user_id = row["user_id"]

        username = row["username"] if "username" in users.columns else str(user_id)

        graph.add_node(
            user_id,
            username=username,
            cluster=int(cluster_lookup.get(user_id, -1))
        )

    filtered_edges = edges[edges["final_similarity"] >= threshold]

    for _, row in filtered_edges.iterrows():
        graph.add_edge(
            row["user_1"],
            row["user_2"],
            weight=float(row["final_similarity"])
        )

    return graph


def visualize_graph(graph, output_path=None, title=None):
    """
    Visualize the user graph.

    Node color = cluster
    Edge thickness = similarity strength
    """

    if title is None:
        title = "Detected Echo Chambers Based on User Similarity"

    pos = nx.spring_layout(
        graph,
        weight="weight",
        seed=42,
        k=0.6
    )

    clusters = nx.get_node_attributes(graph, "cluster")

    node_colors = [
        clusters[node] if clusters[node] != -1 else -1
        for node in graph.nodes()
    ]

    edge_widths = [
        graph[u][v]["weight"] * 5
        for u, v in graph.edges()
    ]

    labels = nx.get_node_attributes(graph, "username")

    plt.figure(figsize=(14, 10))

    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        alpha=0.35
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        cmap=plt.cm.Set3,
        node_size=650,
        edgecolors="black",
        linewidths=0.7
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=8
    )

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)

    plt.show()