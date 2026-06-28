
# Detecting shifts in ideological polarization using NLP-based sentiment analysis pre- and post-identity politics legislation

## Overview

This project detects potential echo chambers and polarization patterns in social media data using graph analysis, semantic similarity, and unsupervised machine learning.

The project builds a user-user network where:

- nodes represent users
- edges represent interactions or similarity between users
- edge weights represent relationship strength
- clusters represent possible echo chambers

## Research Question

Can hidden echo chambers and polarization patterns be detected by combining social network structure, user interaction behavior, and semantic content similarity?

## Dataset

This repository includes a synthetic test dataset based on a fictional policy event:

**Policy:** Digital Integrity Act  
**Policy date:** 2026-03-01

The dataset includes:

- users
- follows
- posts
- comments
- likes
- comment likes
- graph-ready user interaction edges
- before-policy and after-policy labels
- ground-truth synthetic clusters

Because the dataset is synthetic, it can be safely used for testing and demonstration.

## Methodology

The project uses:

- SQL-style relational data
- user-user graph construction
- TF-IDF semantic similarity
- DBSCAN clustering
- NetworkX graph visualization
- structural polarization
- modularity
- semantic polarization
- cross-cluster exposure

## Repository Structure

```text
notebooks/        Jupyter notebooks
data/synthetic/   Synthetic sample dataset
src/              Modular Python code
scripts/          Command-line scripts
outputs/          Generated results
docs/             Methodology notes


````markdown
# Running the Project

## Requirements

Before running the project, ensure that the following software is installed:

- Python 3.10 or later
- Git
- pip (Python package manager)

---

There are two separate programmes for this project
    1) Echo chamber polarization uses offline data with SQL
    2) Clustering on OnlineData YT uses YouTube API (Codes made by Florien de Graaff)


## Instructions for Running Echo Chamber Polarization

## 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/meyrub/echo-chamber-polarization.git
cd echo-chamber-polarization
```

---

## 2. Create a Virtual Environment (Recommended)

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Required Packages

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## 4. Prepare the Dataset

Place the synthetic dataset inside the following directory:

```
data/synthetic/
```

The folder should contain:

```
users.csv
follows.csv
posts.csv
comments.csv
likes.csv
comment_likes.csv
policy_events.csv
user_interaction_edges.csv
daily_summary.csv
synthetic_echo_chamber_policy_dataset.sqlite
```

---

## 5. Verify the Dataset

Before running the analysis, verify that all dataset files are present:

```bash
python scripts/check_dataset.py
```

Successful output will display the number of records loaded from each table.

---

## 6. Run the Analysis Pipeline

Execute the complete clustering and network analysis pipeline:

```bash
python scripts/run_synthetic_pipeline.py
```

The pipeline performs the following tasks:

- Loads the synthetic dataset
- Constructs user similarity features
- Calculates semantic similarity using TF-IDF
- Computes interaction-based similarity metrics
- Generates the user distance matrix
- Detects communities using DBSCAN clustering
- Calculates polarization metrics
- Builds the interaction graph using NetworkX
- Produces a network visualization using Matplotlib
- Saves the analysis results

---

## 7. Output Files

After the analysis has completed, the following files will be generated:

```
outputs/
├── graphs/
│   └── synthetic_echo_chamber_graph.png
│
└── tables/
    ├── detected_user_clusters.csv
    └── cluster_summary.csv
```

The generated network graph visualizes the detected user communities, while the CSV files contain the cluster assignments and summary statistics.

## Instructions for Running Clustering on OnlineData YT
1) Acquire a Google/Youtube API Key
2) Paste it in collect_comments.py for API_KEY
3) Run run_youtube_clustering.py

---

## Troubleshooting

### ModuleNotFoundError

Install the required packages:

```bash
pip install -r requirements.txt
```

---

### FileNotFoundError

Ensure that the synthetic dataset is located in:

```
data/synthetic/
```

---

### Python Not Found

Try one of the following commands:

```bash
python3 scripts/run_synthetic_pipeline.py
```

or on Windows:

```powershell
py scripts\run_synthetic_pipeline.py
```

---


````
