import pandas as pd
import os

# Load the CSV data into a DataFrame
csv_file = '../dataset/train.csv' # Replace with your actual CSV file path
df = pd.read_csv(csv_file)

# Create a directory to store the separated files
output_dir = 'grouped_data'
os.makedirs(output_dir, exist_ok=True)

# Group the data by 'group_id'
grouped = df.groupby('group_id')

# Iterate through each group and save it as a separate CSV file
for group_id, group_data in grouped:
    output_file = os.path.join(output_dir, f'group_{group_id}.csv')
    group_data.to_csv(output_file, index=False)
    print(f'Saved data for group {group_id} to {output_file}')

print("Data separated by group_id and saved successfully.")
