import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from scipy.signal import resample

# def resample_aug(x,is_train=True,target_sr=50):
#     """
#     Resample input x from its original sampling rate to target_sr.
#     Input shape: (T, C), e.g., (100, 3)
#     Output shape: (new_T, C), e.g., (50, 3) if target_sr is 50

#     Args:
#         x (torch.Tensor): input tensor of shape (T, C)
#         is_train (bool): flag for whether augmentation is applied (optional use)
#         target_sr (int): target sampling rate

#     Returns:
#         torch.Tensor: resampled tensor of shape (new_T, C)
#     """
#     T, C = x.shape
#     new_T = int(T * target_sr / 100)
#     sample_X = resample(x, new_T, axis=0)

#     if is_train:
#         sample_X = data_aug(sample_X)
    
#     return sample_X

def data_aug(x):
    """
    Input:
        x: numpy array of shape (T, C), typically (100, 3)
           T is the number of time steps, C is the number of channels
    Output:
        x_aug: numpy array of the same shape with augmentations applied

    Augmentations [1]:
        - Channel permutation
        - Gaussian noise (jittering)
        - Global scaling
        - Segment permutation
    
    Notes: Should not apply normalization for activity data [2]

    Reference:
        [1] https://shamilmamedov.com/blog/2023/da-time-series/
        [2] https://www.mdpi.com/1999-5903/12/11/194
    """

    x = x.astype(np.float32).copy()

    # channel permutation
    perm = np.random.permutation(x.shape[1])
    x = x[:, perm]

    # jittering
    x += np.random.normal(loc=0.0, scale=0.05, size=x.shape)

    # scaling
    x *= np.random.normal(loc=1.0, scale=0.1)

    # segment permute
    seg_len =  x.shape[0] // 4 # 100//4
    segments = np.split(x[:seg_len * 4, :], 4, axis=0)
    perm = np.random.permutation(4)
    x = np.concatenate([segments[i] for i in perm], axis=0)

    return x



import h5py
class iWatch_HDf5(Dataset):
    def __init__(self,
                 root='/niddk-data-central/iWatch/pre_processed_seg/H',
                 set_type='train',
                 transform=None,
                 subset=None):
        self.file_path = os.path.join(root, f"10s_{set_type}.h5")
        self.transform = transform
        # these will be set in the worker when first accessed
        self.h5_file = None
        self.x_data = None
        self.y_data = None
        self.subset= subset
    def _ensure_open(self):
        # called inside worker on first __getitem__
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')
            self.x_data = self.h5_file['x']
            self.y_data = self.h5_file['y']

    def __len__(self):
        # we open here if not already, so that len() works in main process
        self._ensure_open()
        if self.subset is not None:
            return self.subset
        
        return len(self.x_data)

    def __getitem__(self, idx):
        self._ensure_open()                     # open once per worker
        x = self.x_data[idx]                    # shape: (100, 3)
        if self.transform is not None:
            x = self.transform(x) # shape: (100, 3)
        
        x = torch.from_numpy(x).permute(1, 0)  # shape: (3, 100)
        x = x.unsqueeze(0)                      # shape: (1, 3, 100)

        y = int(self.y_data[idx])               
        return x, torch.tensor(y, dtype=torch.long)

    def __del__(self):
        if getattr(self, 'h5_file', None) is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass


def collate_fn(batch):
    clean_batch = []
    for x, y in batch:
        if torch.isnan(x).any() or torch.isinf(x).any():
            continue
        clean_batch.append((x, y))

    if len(clean_batch) == 0:
        return None  # or raise an error if needed

    xs, ys = zip(*clean_batch)
    return torch.stack(xs), torch.tensor(ys)

if __name__ == "__main__":
    print("Starting dataset loading and testing...")

    # Parameters
    batch_size = 4

    train_dataset = iWatch(set_type='train', transform=data_aug)
    test_dataset = iWatch(set_type='test', transform=None)
    val_dataset = iWatch(set_type='val', transform=None)

    # Print the length of the datasets
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("DataLoaders created successfully.")

    # Iterate through the train DataLoader
    print("Training DataLoader:")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        print(f"Images shape: {images.shape}") # bs x nvar x 1 x L
        print(f"Labels shape: {labels.shape}") # bs 
        if i == 1:  # Just show first two batches
            break

    # Iterate through the test DataLoader
    print("\nTesting DataLoader:")
    for i, (images, labels) in enumerate(test_loader):
        print(f"Batch {i + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        if i == 1:  # Just show first two batches
            break