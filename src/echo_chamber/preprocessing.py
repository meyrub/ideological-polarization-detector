import pandas as pd


def build_user_documents_from_posts(posts):
    """
    Build one text document per user from cleaned post data.

    Expected columns:
    - user_id
    - text

    Returns
    -------
    pandas.DataFrame
        DataFrame with user_id and text_profile.
    """

    required_columns = {"user_id", "text"}

    missing = required_columns - set(posts.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    user_documents = (
        posts
        .dropna(subset=["text"])
        .groupby("user_id")["text"]
        .apply(lambda texts: " ".join(texts.astype(str)))
        .reset_index()
        .rename(columns={"text": "text_profile"})
    )

    return user_documents


def add_hashtag_text(posts):
    """
    Combine text and hashtag terms into one analysis field.

    This helps semantic similarity detect topical overlap.
    """

    posts = posts.copy()

    if "hashtag_terms" not in posts.columns:
        posts["hashtag_terms"] = ""

    posts["analysis_text"] = (
        posts["text"].fillna("").astype(str)
        + " "
        + posts["hashtag_terms"].fillna("").astype(str)
    )

    return posts