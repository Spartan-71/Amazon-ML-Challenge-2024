import csv
from collections import defaultdict

# Path to the CSV file
csv_file_path = './dataset/train.csv'

# Dictionary to store grouped data by group_id
grouped_data = defaultdict(list)

# Read the CSV file
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        image_link = row['image_link']
        group_id = row['group_id']
        entity_name = row['entity_name']
        entity_value = row['entity_value']
        grouped_data[group_id].append({
            "image_link": image_link,
            "entity_name": entity_name,
            "entity_value": entity_value
        })

# Printing the grouped data
for group_id, items in grouped_data.items():
    print(f"\nGroup ID: {group_id}")
    for item in items:
        print(f"Image: {item['image_link']}")
        # print(f"  Entity: {item['entity_name']}")
        # print(f"  Value: {item['entity_value']}")
