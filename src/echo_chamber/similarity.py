import numpy as np
import pandas as pd


def filter_edges(edges, threshold=0.35):
    """
    Keep only edges above a similarity threshold.

    Parameters
    ----------
    edges : pandas.DataFrame
        Edge table with final_similarity column.
    threshold : float
        Minimum similarity score required to keep an edge.

    Returns
    -------
    pandas.DataFrame
        Filtered edge table.
    """

    if "final_similarity" not in edges.columns:
        raise ValueError("edges must contain a 'final_similarity' column")

    return edges[edges["final_similarity"] >= threshold].copy()


def build_similarity_matrix(users, edges):
    """
    Build a square user-user similarity matrix from an edge list.

    Parameters
    ----------
    users : pandas.DataFrame
        User table with user_id column.
    edges : pandas.DataFrame
        Edge table with user_1, user_2, and final_similarity columns.

    Returns
    -------
    tuple
        user_ids, similarity_matrix
    """

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

        similarity = float(row["final_similarity"])

        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

    return user_ids, similarity_matrix


def build_distance_matrix(similarity_matrix):
    """
    Convert similarity matrix into distance matrix.

    Formula:
        distance = 1 - similarity
    """

    distance_matrix = 1 - similarity_matrix

    np.fill_diagonal(distance_matrix, 0.0)

    return distance_matrix