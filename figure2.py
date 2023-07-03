import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

score_column = 'Score (Peter)'  # change to Score (0-5) once we have met and discussed
rank_column = 'Rank'
data_dir = 'data'

file1_name = 'Table 1. ChatGPT4 Diagnosis - text without discussion - txt_cases_results.tsv'
file2_name = 'Table 2. ChatGPT4 Diagnosis - only age_sex_signs_symptoms - phenopacket_based_queries_results.tsv'

file1_label = 'Narrative-based queries'
file2_label = 'Feature-based queries'

file1 = pd.read_csv(os.path.join(data_dir, file1_name), sep="\t", header=0)
file2 = pd.read_csv(os.path.join(data_dir, file2_name), sep="\t", header=0)

file1_score = file1[score_column]
file2_score = file2[score_column]

# Create a new dataframe with the extracted columns
df = pd.DataFrame({
    "file1_score": file1_score,
    "file2_score": file2_score
})


def to_numeric(s: pd.Series) -> pd.Series:
    # Convert non-numeric values to NaN
    numeric_series = pd.to_numeric(s, errors='coerce')
    # Filter out non-numeric values
    return s[numeric_series.notna()]


# Calculate the counts of each score (0 to 5) in both columns
score_counts_file1 = to_numeric(df['file1_score']).value_counts().sort_index()
score_counts_file2 = to_numeric(df['file2_score']).value_counts().sort_index()

# Create the first grouped histogram
fig1, ax1 = plt.subplots()
bar_width = 0.35
index = score_counts_file1.index

bar1 = ax1.bar(index - bar_width/2, score_counts_file1, bar_width, label=file1_label)
bar2 = ax1.bar(index + bar_width/2, score_counts_file2, bar_width, label=file2_label)

ax1.set_xlabel('Score')
ax1.set_ylabel('Count')
ax1.set_title('A')
ax1.set_xticks(index)
ax1.set_xticklabels(index)
ax1.legend()

# Save the first figure to a PNG file
fig1.savefig('figure2a.png')

# Create the second grouped histogram
fig2, ax2 = plt.subplots()

file1_rank = file1[rank_column]
file2_rank = file2[rank_column]

# Create a new dataframe with the extracted columns
df = pd.DataFrame({
    "file1_score": file1_score,
    "file1_rank": file1_rank,
    "file2_score": file2_score,
    "file2_rank": file2_rank
})

def set_empty_indices_to_zero(series: pd.Series, this_min: int,
                              this_max: int) -> pd.Series:
    # Create a new series with the desired index range
    new_series = pd.Series(index=range(this_min, this_max + 1), dtype=series.dtype)

    # Update the new series with values from the original series
    new_series.update(series)

    # Fill missing indices with zeros
    new_series = new_series.fillna(0)

    return new_series.sort_index()


# Find rank where score is 4 or 5, then count the number of times each rank appears
rank_counts_file1 = to_numeric(df[df['file1_score'].isin([4, 5])]['file1_rank']).value_counts().sort_index()
rank_counts_file2 = to_numeric(df[df['file2_score'].isin(["4", "5"])]['file2_rank']).value_counts().sort_index()

this_max = int(max(rank_counts_file1.index.tolist() + rank_counts_file2.index.tolist()))
this_min = int(max(min(rank_counts_file1.index.tolist() + rank_counts_file2.index.tolist()), 1))

rank_counts_file1 = set_empty_indices_to_zero(rank_counts_file1, this_min, this_max)
rank_counts_file2 = set_empty_indices_to_zero(rank_counts_file2, this_min, this_max)

bar3 = ax2.bar(rank_counts_file1.index - bar_width/2, rank_counts_file1, bar_width, alpha=1, label=file1_label)
bar4 = ax2.bar(rank_counts_file2.index + bar_width/2, rank_counts_file2, bar_width, alpha=1, label=file2_label)

ax2.set_xlabel('Rank')
ax2.set_ylabel('Count')
ax2.set_title('B')
ax2.set_xticks(rank_counts_file1.index)
ax2.set_xticklabels(rank_counts_file1.index)
ax2.legend()

# Save the second figure to a PNG file
fig2.savefig('figure2b.png')

# Combine both figures side by side into a single PNG file
combined_fig = plt.figure(figsize=(12, 6))
combined_fig.add_subplot(1, 2, 1)
combined_fig.add_subplot(1, 2, 2)

img1 = plt.imread('figure2a.png')
img2 = plt.imread('figure2b.png')

plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.axis('off')

# Save the combined figure to a PNG file
combined_fig.savefig('combined_figure.png')
