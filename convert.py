import pandas as pd

# Load the first CSV file (category, body, title)
file1 = 'news-article-categories.csv'  # Ganti dengan nama file Anda

# Read the first CSV file
data1 = pd.read_csv(file1)

# Create the second CSV structure manually
columns_file2 = ['title', 'content', 'category', 'tags']
data2 = pd.DataFrame(columns=columns_file2)

# Map columns from file1 to match file2
data1_mapped = data1.rename(columns={
    'category': 'category',
    'body': 'content',
    'title': 'title'
})

# Add a default 'tags' column for file1 data
data1_mapped['tags'] = ''

# Ensure both dataframes have the same columns
data1_mapped = data1_mapped[data2.columns]

# Combine the two datasets
combined_data = pd.concat([data2, data1_mapped], ignore_index=True)

# Save the combined dataset to a new CSV file
output_file = 'merged_file.csv'  # Ganti dengan nama file output Anda
combined_data.to_csv(output_file, index=False)

print(f"Data berhasil digabungkan dan disimpan dalam {output_file}")
