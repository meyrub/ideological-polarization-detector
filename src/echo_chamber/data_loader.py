import pandas as pd


def load_project_csvs(users_path, posts_path, edges_path, hashtags_path=None):
    """
    Load the project-ready CSV files.

    Parameters
    ----------
    users_path : str or Path
        Path to the user/node table.
    posts_path : str or Path
        Path to the cleaned posts table.
    edges_path : str or Path
        Path to the inferred user-user edge table.
    hashtags_path : str or Path, optional
        Path to the post-hashtag table.

    Returns
    -------
    dict
        Dictionary containing loaded DataFrames.
    """

    users = pd.read_csv(users_path)
    posts = pd.read_csv(posts_path)
    edges = pd.read_csv(edges_path)

    data = {
        "users": users,
        "posts": posts,
        "edges": edges,
    }

    if hashtags_path is not None:
        hashtags = pd.read_csv(hashtags_path)
        data["hashtags"] = hashtags

    return data