import os
import shutil
import pandas as pd

# Open this csv: /niddk-data-central/iWatch/support_files/iWatch_randomization.csv
csv_path = '/niddk-data-central/iWatch/support_files/iWatch_randomization.csv'
train_test_split = pd.read_csv(csv_path)
train_ids = train_test_split[train_test_split['type'] == 'train_set']['ID'].tolist()
test_ids = train_test_split[train_test_split['type'] == 'test_set']['ID'].tolist()

# Define source and destination directories
source_dir = '/niddk-data-central/iWatch/pre_processed_pt/W'
train_dest_dir = '/niddk-data-central/iWatch/pre_processed_test/H/train'
test_dest_dir = '/niddk-data-central/iWatch/pre_processed_test/H/test'

# Create destination directories if they don't exist
os.makedirs(train_dest_dir, exist_ok=True)
os.makedirs(test_dest_dir, exist_ok=True)

# Move train IDs
for train_id in train_ids:
    src_path = os.path.join(source_dir, train_id)
    dest_path = os.path.join(train_dest_dir, train_id)
    if os.path.exists(src_path):
        shutil.copytree(src_path, dest_path)

# Move test IDs
for test_id in test_ids:
    src_path = os.path.join(source_dir, test_id)
    dest_path = os.path.join(test_dest_dir, test_id)
    if os.path.exists(src_path):
        shutil.copytree(src_path, dest_path)