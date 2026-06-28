import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


df = pd.read_csv("youtube_comments.csv")
df = df.dropna(subset=["comment"]).copy()
df["comment"] = df["comment"].astype(str)

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=1000,
    min_df=2
)

X = vectorizer.fit_transform(df["comment"])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.toarray())

df["x"] = X_2d[:, 0]
df["y"] = X_2d[:, 1]

plt.figure(figsize=(10, 7))

for cluster in sorted(df["cluster"].unique()):
    subset = df[df["cluster"] == cluster]

    plt.scatter(
        subset["x"],
        subset["y"],
        label=f"Cluster {cluster}",
        alpha=0.75,
        s=55
    )

plt.title("Unsupervised clustering of YouTube comments")
plt.xlabel("Text similarity dimension 1")
plt.ylabel("Text similarity dimension 2")
plt.legend()
plt.tight_layout()

plt.savefig("clean_comment_clusters.png", dpi=300)
plt.show()

df.to_csv("comment_cluster_results.csv", index=False)

print(df["cluster"].value_counts().sort_index())
print(df.groupby(["cluster", "period"]).size())