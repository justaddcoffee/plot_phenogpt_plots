import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

score_column = 'Score (Peter)'  # change to Score (0-5) once we have met and discussed
data_dir = 'data'

file1_name = 'Table 1. ChatGPT4 Diagnosis - text without discussion - txt_cases_results.tsv'
file2_name = 'Table 2. ChatGPT4 Diagnosis - only age_sex_signs_symptoms - phenopacket_based_queries_results.tsv'

file1_label = 'Narative-based queries'
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

# Create a grouped histogram
fig, ax = plt.subplots()
bar_width = 0.35
index = score_counts_file1.index

bar1 = ax.bar(index - bar_width/2, score_counts_file1, bar_width, label=file1_label)
bar2 = ax.bar(index + bar_width/2, score_counts_file2, bar_width, label=file2_label)

ax.set_xlabel('Score')
ax.set_ylabel('Count')
ax.set_title('Comparison of scores for narrative-based and feature-based GPT queries')
ax.set_xticks(index)
ax.set_xticklabels(index)
ax.legend()

plt.show()

# write out plot to file
fig.savefig('figure2a.png')
