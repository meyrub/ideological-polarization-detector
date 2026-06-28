import pandas as pd
from sklearn.cluster import DBSCAN


def cluster_users_dbscan(user_ids, distance_matrix, eps=0.65, min_samples=2):
    """
    Cluster users with DBSCAN using a precomputed distance matrix.

    Parameters
    ----------
    user_ids : list
        List of user IDs.
    distance_matrix : numpy.ndarray
        Square user-user distance matrix.
    eps : float
        Maximum distance between users for them to be neighbors.
    min_samples : int
        Minimum number of users required to form a cluster.

    Returns
    -------
    pandas.DataFrame
        DataFrame with user_id and cluster.
    """

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed"
    )

    labels = model.fit_predict(distance_matrix)

    return pd.DataFrame({
        "user_id": user_ids,
        "cluster": labels
    })