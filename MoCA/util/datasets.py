import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
import torch


# Helper function, load numpy that from later version of numpy
class LegacyNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect deprecated or removed internal numpy modules
        if module.startswith("numpy._core"):
            module = module.replace("_core", "core")
        if module == "numpy.core.multiarray":
            import numpy.core.multiarray
            return getattr(numpy.core.multiarray, name)
        return super().find_class(module, name)

def data_aug(x):
    '''
    input: x: (nvar, L)

    Jittering: add small Gaussian noise to each axis to simulate measurement error.

    Scaling: multiply the full sequence by a random factor (e.g. between 0.9 and 1.1) to mimic signal‐strength variation.

    Rotation: apply a random rotation in the horizontal plane (x–y axes) to reflect changes in device orientation.

    Permutation: split the series into equal‐length segments and shuffle their order to break global dependencies.

    '''

    # Jittering: add Gaussian noise with std = 0.02
    if torch.rand(1).item() < 0.5:
        x = x + torch.randn_like(x) * 0.02

    # Scaling: multiply by a random factor in [0.9, 1.1]
    if torch.rand(1).item() < 0.5:
        factor = torch.rand(1).item() * 0.2 + 0.9
        x = x * factor

    # Rotation: rotate x–y axes of both accel and gyro
    if x.size(0) >= 3 and torch.rand(1).item() < 0.5:
        theta = torch.rand(1) * (torch.pi/2) - (torch.pi/4)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        x_rot = x.clone()
        # accelerate channels 0,1
        x0, x1 = x[0], x[1]
        x_rot[0] = cos_t * x0 - sin_t * x1
        x_rot[1] = sin_t * x0 + cos_t * x1
    
        # if x.size(0) >= 6:
        #     # gyroscope channels 3,4
        #     g0, g1 = x[3], x[4]
        #     x_rot[3] = cos_t * g0 - sin_t * g1
        #     x_rot[4] = sin_t * g0 + cos_t * g1
        #     x = x_rot

    # Permutation: split into 4 segments and shuffle
    if torch.rand(1).item() < 0.5:
        nvar, L = x.shape
        n_seg = 4
        seg_len = L // n_seg
        segments = [x[:, i*seg_len:(i+1)*seg_len] for i in range(n_seg)]
        perm = torch.randperm(n_seg)
        x = torch.cat([segments[i] for i in perm], dim=1)

    return x


class iWatch(Dataset):
    def __init__(self, 
                root='/niddk-data-central/iWatch/pre_processed_seg/H', 
                set_type='train',
                transform=None):
        
        self.root = root
        self.set_type = set_type
        # Set file paths
        if set_type == 'train':
            self.data_path = os.path.join(self.root, 'train')
        elif set_type == 'val':
            self.data_path = os.path.join(self.root, 'val')
        elif set_type == 'test':
            self.data_path = os.path.join(self.root, 'test')
        else:
            raise ValueError("set_type must be 'train', 'val', or 'test'")

        self.transform = transform

    def __len__(self):
        return 1733088 # H 1733088, W 1785840

    def __getitem__(self, idx):
        # load the data
        fn = os.path.join(self.data_path, f"{idx}.pkl")
        with open(fn, "rb") as f:
            #data = pickle.load(f)
            data = LegacyNumpyUnpickler(f).load()

        x = data['x']  # np.array shape: (100, 3)

        # Normalization
        x = torch.from_numpy(x.transpose(1, 0)).to(torch.float32)
        x = x / x.abs().mean()  # (3,100)

        if self.transform is not None:
            x = self.transform(x) # (3,100) tensor

        sample_y = data['y']  # np.int
        sample_y = torch.tensor(sample_y, dtype=torch.long)

        sample_X = x.unsqueeze(0)  # (1,3,100)                

        return sample_X, sample_y


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