import os
import pickle
import torch
from commons import get_dataloaders
from einops import rearrange
from tqdm import tqdm

def save_samples_from_loader(dataloader, out_dir):
    """
    Iterate through a DataLoader and save each sample as a pickle.
    Each file is named 0.pkl, 1.pkl, … and contains {'x', 'y', 'fn'}.
    """
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    for batch in tqdm(dataloader, desc="Processing batches"):
        # assume batch is (x_batch, y_batch)
        x_batch, y_batch = batch
        print(f"Batch shape: {x_batch.shape}, {y_batch.shape}") # torch.Size([BS, win_size, 100, 3]) torch.Size([16, 42]) 
        x_batch = rearrange(x_batch, "b t c l -> (b t) c l")
        y_batch = y_batch.view(-1)  # flatten y_batch
        

        # if tensors, move to cpu and convert to numpy
        if torch.is_tensor(x_batch):
            x_batch = x_batch.cpu().numpy()
        if torch.is_tensor(y_batch):
            y_batch = y_batch.cpu().numpy()

        # handle batched data
        for i in range(x_batch.shape[0]):
            sample = {
                "x": x_batch[i],
                "y": y_batch[i],
            }
            file_path = os.path.join(out_dir, f"{idx}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(sample, f)
            idx += 1


if __name__ == "__main__":
    split_data_file = "/niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl"
    pre_processed_dir = '/niddk-data-central/iWatch/pre_processed_pt/H'  # '/niddk-data-central/iWatch/pre_processed_pt/W'

    with open(split_data_file, "rb") as f:
        split_data = pickle.load(f)

        train_subjects = split_data["train"]
        valid_subjects = split_data["val"]
        test_subjects = split_data["test"]

            
    train_dl, valid_dl, test_dl = get_dataloaders(
        pre_processed_dir=pre_processed_dir,
        bi_lstm_win_size=42, # CHAP_Adult: 60 // 10 * 7 = 42
        batch_size=16, 
        train_subjects=train_subjects, 
        valid_subjects=valid_subjects, 
        test_subjects=test_subjects,
    )

    if train_dl is not None:
        save_samples_from_loader(train_dl, "/niddk-data-central/iWatch/pre_processed_seg/H/train")
    if valid_dl is not None:
        save_samples_from_loader(valid_dl, "/niddk-data-central/iWatch/pre_processed_seg/H/val")
    if test_dl is not None:
        save_samples_from_loader(test_dl, "/niddk-data-central/iWatch/pre_processed_seg/H/test")
    
    # print how many files in each folder
    for split in ["train", "val", "test"]:
        path = os.path.join("/niddk-data-central/iWatch/pre_processed_seg/H", split)
        num_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        print(f"Number of files in {split}: {num_files}")
    print('Done!')
   
# python -m data_seg