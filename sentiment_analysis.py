import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Load comments
df = pd.read_csv("youtube_comments.csv")

# Load sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Function to classify one comment
def classify_sentiment(text):
    try:
        result = sentiment_model(str(text)[:512])[0]
        return result["label"], result["score"]
    except Exception:
        return "unknown", 0

# Apply model
labels = []
scores = []

for comment in df["comment"]:
    label, score = classify_sentiment(comment)
    labels.append(label)
    scores.append(score)

df["sentiment"] = labels
df["confidence"] = scores

# Save full results
df.to_csv("sentiment_results.csv", index=False)

# Count sentiment by before/after period
summary = (
    df.groupby(["period", "sentiment"])
    .size()
    .reset_index(name="count")
)

print(summary)

# Convert to percentages
percentage_summary = (
    df.groupby("period")["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("percentage")
    .reset_index()
)

print(percentage_summary)

# Plot
pivot = percentage_summary.pivot(
    index="period",
    columns="sentiment",
    values="percentage"
).fillna(0)

# Force before to appear first
pivot = pivot.reindex(["before", "after"])

pivot.plot(kind="bar")

plt.title("Sentiment before and after policy event")
plt.xlabel("Period")
plt.ylabel("Percentage of comments")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sentiment_plot.png")
plt.show()