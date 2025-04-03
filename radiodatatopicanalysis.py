import pandas as pd
from collections import defaultdict, Counter
import re

# Load dataset
dataset_path = "/content/AZ.parquet"
df = pd.read_parquet(dataset_path)

# Manually curated topic dictionary
topic_dict = {
    "abortion": [
        "birth control", "contraception", "abortion provider", "ACOG",
        "medication abortion", "medical abortion", "planned parenthood", "pro-choice", "pro-life"
    ],
    "international affairs": [
        "russia", "ukraine", "israel", "gaza",
        "who involvement", "climate agreements", "foreign policy", "international relations"
    ],
    "immigration": [
        "border control", "citizenship", "immigration policy", "border security",
        "deportation", "visa", "migrants", "asylum"
    ],
    "economy": [
        "job opportunities", "jobs", "tariffs", "trade", "inflation", "tax",
        "budget deficit", "unemployment", "economic growth"
    ],
    "violent crime": [
        "gun control", "school shootings", "2nd amendment", "second amendment",
        "firearms", "mass shooting", "gun violence", "crime rate"
    ],
    "climate change": [
        "science", "environment", "disaster", "climate crisis", "global warming",
        "carbon emissions", "green energy", "natural disasters"
    ]
}

# Store results
results = []

# Loop through topics
for topic, keywords in topic_dict.items():
    total_mentions = 0
    keyword_counts = Counter()

    for text in df['text'].dropna():
        text_lower = text.lower()

        matched = False
        for kw in keywords:
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                matched = True
                keyword_counts[kw] += len(matches)

        if matched:
            total_mentions += 1

    results.append({
        "Topic": topic,
        "Total Mentions": total_mentions,
        "Keyword Frequency": dict(keyword_counts)
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Show result
display(results_df)

# Save to file
results_df.to_csv("/content/topic_mentions_summary.csv", index=False)
print("Results saved to: /content/topic_mentions_summary.csv")
