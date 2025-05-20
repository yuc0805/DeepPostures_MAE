import os
from commons import get_dataloaders

#TODO: find hip, need to find wrist
data_path = "/niddk-data-central/iWatch/pre_processed_pt/H"

with open("/niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl", "rb") as f:
    split_data = pickle.load(f)

train_subjects = split_data["train"]
valid_subjects = None #split_data["val"]


data_loader_train, data_loader_val, _ = get_dataloaders(
pre_processed_dir=args.data_path,
bi_lstm_win_size=42, # chap_adult
batch_size=32,
train_subjects=train_subjects,
valid_subjects=valid_subjects,
test_subjects=None,)

positive_count = 0
negative_count = 0
for inputs, labels in data_loader_train:
    # input: (bs, 42, 100, 3)
    
    labels = labels.view(-1)

    # calculate the pos_weight
    positive_count += (labels == 1).sum().item() # non-sitting
    negative_count += (labels == 0).sum().item() # sitting, this dominates

pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
print(f"positive_count: {positive_count}")
print(f"negative_count: {negative_count}")
print(f"pos_weight: {pos_weight}")

