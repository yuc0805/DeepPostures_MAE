import os
import pickle
import torch
from commons import get_dataloaders

def save_samples_from_loader(dataloader, out_dir):
    """
    Iterate through a DataLoader and save each sample as a pickle.
    Each file is named 0.pkl, 1.pkl, … and contains {'x', 'y', '.
    """
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    for batch in dataloader:
        # assume batch is (x_batch, y_batch)
        x_batch, y_batch = batch
        print(f"Batch shape: {x_batch.shape}, {y_batch.shape}")

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
    with open(split_data_file, "rb") as f:
        split_data = pickle.load(f)

        train_subjects = split_data["train"]
        valid_subjects = split_data["val"]
        test_subjects = split_data["test"]

            
    train_dl, valid_dl, test_dl = get_dataloaders(
        pre_processed_dir="path/to/preproc",
        bi_lstm_win_size=7, # CHAP_Adult
        batch_size=16, 
        train_subjects=train_subjects, 
        valid_subjects=valid_subjects, 
        test_subjects=test_subjects,
    )

    if train_dl is not None:
        save_samples_from_loader(train_dl, "train_samples")
    if valid_dl is not None:
        save_samples_from_loader(valid_dl, "valid_samples")
    if test_dl is not None:
        save_samples_from_loader(test_dl, "test_samples")
