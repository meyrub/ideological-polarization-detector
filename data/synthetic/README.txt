Synthetic Echo Chamber / Polarization Policy Dataset
====================================================

This is a synthetic test dataset designed for a social media echo chamber and
polarization project.

Policy event:
- Policy name: Digital Integrity Act
- Policy date: 2026-03-01
- Before window: 2025-12-31 to 2026-02-28
- After window: 2026-03-02 to 2026-04-30

Main purpose:
This dataset lets you test whether your project can:
1. build user-user graphs,
2. detect hidden groups / echo chambers,
3. compare before-policy and after-policy behavior,
4. measure polarization,
5. evaluate clustering results against ground-truth synthetic labels.

Files:
- users.csv
- follows.csv
- posts.csv
- comments.csv
- likes.csv
- comment_likes.csv
- policy_events.csv
- user_interaction_edges.csv
- daily_summary.csv
- synthetic_echo_chamber_policy_dataset.sqlite

Counts:
{'users': 360, 'follows': 8927, 'posts': 2400, 'comments': 6552, 'likes': 23703, 'comment_likes': 22339, 'interaction_edges': 25966, 'daily_summary_rows': 121}

Ground-truth user clusters:
{'C3_privacy_skeptics': 72, 'C1_pro_policy': 100, 'C4_neutral_news': 40, 'C2_anti_policy': 99, 'C5_general_public': 49}

Post timing:
- before-policy posts: 1169
- policy-day posts: 12
- after-policy posts: 1219

Important design feature:
The dataset intentionally increases intra-cluster interaction after the policy
passes. This creates a measurable before/after polarization effect.

Interpretation:
- This is synthetic data, not real user data.
- Usernames, posts, comments, likes, and follows are artificial.
- The dataset includes ground_truth_cluster and ground_truth_stance columns for testing.
- In a real project, you would usually not have ground-truth labels.

Recommended use:
- Use users.csv as the node table.
- Use user_interaction_edges.csv as the graph edge table.
- Use posts.csv and comments.csv for semantic analysis.
- Use follows.csv, likes.csv, comments.csv, and comment_likes.csv to test SQL feature extraction.
- Use policy_events.csv to divide the data into before and after periods.
- Use daily_summary.csv for simple time-series visualizations.

Suggested graph threshold:
- final_interaction_weight >= 0.08 for a broad graph
- final_interaction_weight >= 0.12 for a cleaner graph
- final_interaction_weight >= 0.18 for a strict graph

Suggested DBSCAN start:
- eps = 0.80 when using distance = 1 - final_interaction_weight
- min_samples = 3

Note:
Because final_interaction_weight is intentionally sparse and conservative,
you may need to tune thresholds and clustering parameters.