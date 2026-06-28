from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/synthetic")

users = pd.read_csv(DATA_DIR / "users.csv")
posts = pd.read_csv(DATA_DIR / "posts.csv")
comments = pd.read_csv(DATA_DIR / "comments.csv")
likes = pd.read_csv(DATA_DIR / "likes.csv")
edges = pd.read_csv(DATA_DIR / "user_interaction_edges.csv")

print("Dataset loaded successfully.")
print()
print("Users:", len(users))
print("Posts:", len(posts))
print("Comments:", len(comments))
print("Likes:", len(likes))
print("Graph edges:", len(edges))

print()
print("User columns:")
print(users.columns.tolist())

print()
print("Edge columns:")
print(edges.columns.tolist())