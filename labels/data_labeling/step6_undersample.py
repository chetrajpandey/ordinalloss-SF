import pandas as pd
import numpy as np


df1 = pd.read_csv('data/-90_to_90/Fold1_test.csv')
df2 = pd.read_csv('data/-90_to_90/Fold2_test.csv')
df3 = pd.read_csv('data/-90_to_90/Fold3_test.csv')
df4 = pd.read_csv('data/-90_to_90/Fold4_test.csv')

df = pd.concat([df1, df2])
df['goes_class'] = df['goes_class'].replace(np.nan, 'FQ')

# Select rows where goes_class starts with 'M' or 'X'
selected_rows_x = df[df['goes_class'].str.startswith('X')]
selected_rows_m = df[df['goes_class'].str.startswith('M')]

# Select rows where goes_class is 'C', 'B', or empty string
selected_rows_c = df[df['goes_class'].str.startswith('C')]
selected_rows_b = df[df['goes_class'].str.startswith('B')]
selected_rows_a = df[df['goes_class'].str.startswith('A')]
selected_rows_nf = df[df['goes_class'].str.startswith('FQ')]

# print(len(selected_rows_x), len(selected_rows_m), len(selected_rows_c), len(selected_rows_b), len(selected_rows_a), len(selected_rows_nf))

# Calculate the required fraction for 'NF' to balance the data


# Get all instances of 'M' and 'X'
selected_rows_flare = pd.concat([selected_rows_m, selected_rows_x])

# Get 5% random sample from selected_rows_c, selected_rows_b, and selected_rows_a
random_sample_c = selected_rows_c.sample(frac=0.30, random_state=10)
random_sample_b = selected_rows_b.sample(frac=0.30, random_state=10)
random_sample_a = selected_rows_a.sample(frac=0.30, random_state=10)

# Get the remaining instances from 'NF' to balance the data
remaining_nf_count = 6*len(selected_rows_flare) - (len(random_sample_c) + len(random_sample_b) + len(random_sample_a))

print("NF percentage: ", (remaining_nf_count/len(selected_rows_nf)*100))
random_sample_nf = selected_rows_nf.sample(n=remaining_nf_count, random_state=10)


# Apply the custom function to create the 'log_scale' column
# random_sample_nf['log_scale'] = random_sample_nf['goes_class'].apply(map_log_scale)
# print(len(selected_rows_flare), len(random_sample_c) + len(random_sample_b) + len(random_sample_a) + len(random_sample_nf))

# Concatenate the DataFrames
final_result = pd.concat([selected_rows_flare, random_sample_c, random_sample_b, random_sample_a, random_sample_nf])
print(final_result['label'].value_counts())


final_result.to_csv('data/train.csv', index=False, header=True)