import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

score_column = 'Score (Peter)'  # change to Score (0-5) once we have met and discussed
data_dir = 'data'

file1_name = 'Table 1. ChatGPT4 Diagnosis - text without discussion - txt_cases_results.tsv'
file2_name = 'Table 2. ChatGPT4 Diagnosis - only age_sex_signs_symptoms - phenopacket_based_queries_results.tsv'

file1 = pd.read_csv(os.path.join(data_dir, file1_name), sep="\t", header=0)
file2 = pd.read_csv(os.path.join(data_dir, file2_name), sep="\t", header=0)

file1_score = file1[score_column]
file2_score = file2[score_column]

# Create a new dataframe with the extracted columns
df = pd.DataFrame({
    "file1_score": file1_score,
    "file2_score": file2_score
})

import pandas as pd
import matplotlib.pyplot as plt

# Sample data (replace this with your actual data)
data = {
    "file1_score": [1, 3, 0, 2, 1, 5, 3, 0, 4, 1],
    "file2_score": [0, 3, 1, 5, 4, 0, 2, 1, 2, 5]
}

df = pd.DataFrame(data)

# Step 1: Calculate the counts of each score (0 to 5) in both columns
score_counts_file1 = df['file1_score'].value_counts().sort_index()
score_counts_file2 = df['file2_score'].value_counts().sort_index()

# Step 2: Create a grouped histogram
fig, ax = plt.subplots()
bar_width = 0.35
index = score_counts_file1.index

bar1 = ax.bar(index - bar_width/2, score_counts_file1, bar_width, label='File 1')
bar2 = ax.bar(index + bar_width/2, score_counts_file2, bar_width, label='File 2')

ax.set_xlabel('Score')
ax.set_ylabel('Count')
ax.set_title('Grouped Histogram of File 1 and File 2 Scores')
ax.set_xticks(index)
ax.set_xticklabels(index)
ax.legend()

plt.show()

# write out plot to file
fig.savefig('figure2a.png')
