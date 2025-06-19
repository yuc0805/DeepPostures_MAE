import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from scipy.signal import resample
from transforms3d.axangles import axangle2mat
from einops import rearrange

def rotation_axis(sample):
    """

    Rotate the input sample along a random axis by a random angle.
    Modified from: OxWearables. (2022). 
    ssl-wearables: Self-supervised learning for wearable sensor data. 
    GitHub repository. https://github.com/OxWearables/ssl-wearables/blob/main/sslearning/data/datautils.py
    
    Args:
        sample (numpy.ndarray): Input sample of shape (T, C), where T is the number of time steps and C is the number of channels.
    
    Returns:
        numpy.ndarray: Rotated sample of the same shape as input.
    """
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
    rotation_matrix = axangle2mat(axis, angle)
    sample = np.matmul(sample, rotation_matrix)

    return sample

def data_aug(x):
    """
    Input:
        x: numpy array of shape (T, C), typically (100, 3)
           T is the number of time steps, C is the number of channels
    Output:
        x_aug: numpy array of the same shape with augmentations applied

    Augmentations [1]:
        - Gaussian noise (jittering)
        - Global scaling
        - Segment permutation
        - Channel permutation: invariant to the order of channels, because different device manafacturers may have different channel orders
        - Axis flipping: we want the model invariant to subject that wear the device differently
    
    Notes: 
        -Should not apply normalization for activity data [2]


    Reference:
        [1] https://shamilmamedov.com/blog/2023/da-time-series/
        [2] https://www.mdpi.com/1999-5903/12/11/194

    """

    x = x.astype(np.float32).copy()
    # rotation
    x = rotation_axis(x) 

    # channel permutation
    perm = np.random.permutation(x.shape[1])
    x = x[:, perm]
           
    # jittering
    x += np.random.normal(loc=0.0, scale=0.05, size=x.shape)

    # scaling
    x *= np.random.normal(loc=1.0, scale=0.1)

    # segment permute
    # seg_len =  x.shape[0] // 4 # 100//4
    # segments = np.split(x[:seg_len * 4, :], 4, axis=0)
    # perm = np.random.permutation(4)
    # x = np.concatenate([segments[i] for i in perm], axis=0)

    return x

import h5py
class iWatch(Dataset):
    def __init__(self, 
                 root='/niddk-data-central/iWatch/pre_processed_long_seg',
                 set_type='train',
                 transform=None,
                 subset_ratio=1.0):
        
        self.file_path = os.path.join(root, f"10s_{set_type}.h5")
        self.data_file = h5py.File(self.file_path, 'r')
        self.x_data = self.data_file['x']       # shape: (N,window, 100, 3)
        self.y_data = self.data_file['y'] 
        
        self.transform = transform

        
        self.indices = np.arange(len(self.y_data))

        # each subject should have 10% so the distrbution for each subject is the same as before
        if subset_ratio < 1.0:
            self.subject_id = self.data_file['subject_id']
            np.random.seed(42)
            final_indices = []
            subject_ids = np.unique(self.subject_id)
            for sid in subject_ids:
                subject_indices = np.where(self.subject_id[:] == sid)[0]
                num_samples = max(1, int(len(subject_indices) * subset_ratio))
                sampled = np.random.choice(subject_indices, num_samples, replace=False)
                final_indices.extend(sampled)

            self.indices = np.array(final_indices)

    # def resample_epoch(self):
    #     """
    #     Build a new balanced index list by copying the minority label
    #     with replacement until both labels have the same count, then shuffle.
    #     """
    #     labels = np.asarray(self.y_data, dtype=np.int64)
    #     classes, counts = np.unique(labels, return_counts=True)
    #     target = counts.max()                     # majority label size
    #     new_idx = []
    #     for c in classes:
    #         idx_c = self.all_idx[labels == c]
    #         if len(idx_c) < target:
    #             extra = self.rng.choice(idx_c, size=target - len(idx_c), replace=True)
    #             idx_c = np.concatenate([idx_c, extra])
    #         new_idx.append(idx_c)
    #     self.indices = np.concatenate(new_idx)
    #     self.rng.shuffle(self.indices)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        x = self.x_data[idx]  # shape: (42, 100, 3)
        y = self.y_data[idx]  # shape: (42,)

        if self.transform is not None:
            x_aug = x.reshape(-1, x.shape[-1]) # (4200,3)
            x_aug = self.transform(x_aug)
            x_aug = x_aug.reshape(x.shape[0], x.shape[1], -1)  # Reshape back to (42, 100, 3)

        else:
            x_aug = x.copy()

        x_aug = torch.from_numpy(x_aug)
        y = torch.tensor(y, dtype=torch.long)

        return x_aug, y

    
# Dataset for (BS, 100,3)
import h5py
class iWatch_HDf5(Dataset):
    def __init__(self,
                 root='/niddk-data-central/iWatch/pre_processed_seg/H',
                 set_type='train',
                 transform=None,
                 subset=None,
                 target_sr=10):
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