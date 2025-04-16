import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

class UCIHAR(Dataset):
    def __init__(self, data_path, is_test=False, pre_mix_up = False,
                 normalization=False,transform=False,mix_up=True,nb_classes=7):
        # Set file paths

        # only for off-line mix_up in pretrain stage
        if pre_mix_up:
            x_train_dir = os.path.join(data_path, 'X_train_aug_all.pt')
            self.X = torch.tensor(torch.load(x_train_dir),dtype=torch.float32)
            # dummy variable
            self.y = torch.zeros(self.X.shape[0], dtype=torch.long)

        else:
            X_train_dir = os.path.join(data_path, 'X_train_all.pt')
            y_train_dir = os.path.join(data_path, 'y_train_all_mode.pt')
            X_test_dir = os.path.join(data_path, 'X_test_all.pt')
            y_test_dir = os.path.join(data_path, 'y_test_all_mode.pt')

            # Load data based on whether it's a test set or training set
            if is_test:
                self.X = torch.tensor(torch.load(X_test_dir),dtype=torch.float32).permute(0, 3, 1, 2).squeeze()
                self.y = torch.tensor(torch.load(y_test_dir), dtype=torch.long)
            else:
                self.X = torch.tensor(torch.load(X_train_dir),dtype=torch.float32).permute(0, 3, 1, 2).squeeze()
                self.y = torch.tensor(torch.load(y_train_dir), dtype=torch.long)

            if nb_classes==6:
                # Filter out samples with label 0 and adjust labels 1-6 to 0-5
                mask = self.y != 0
                self.X = self.X[mask]
                self.y = self.y[mask] - 1
                print(self.y)

        self.normalization = normalization
        self.transform = transform
        self.mix_up = mix_up

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx]
        sample_y = self.y[idx]

        # Apply normalization if transform is set
        if self.normalization:
            sample_X = (sample_X - sample_X.mean(dim=1,keepdim=True)) / sample_X.std(dim=1,keepdim=True)

        if self.mix_up:
            # Randomly select another sample
            mix_idx = torch.randint(0, len(self.X), (1,)).item()
            sample_X2 = self.X[mix_idx]
            sample_X = self.mix_time_series(sample_X, sample_X2)


        # only for vit_base line
        if self.transform:
            sample_X = sample_X.unsqueeze(0).unsqueeze(0)
            sample_X = F.interpolate(sample_X, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0).repeat(1, 3, 1, 1).squeeze() 
        else:
            sample_X = sample_X.unsqueeze(0) # C,H,W [1,6,200],

        return sample_X, sample_y

    def mix_time_series(self, sample1, sample2):
        # Squeeze to remove singleton dimensions
        sample1 = sample1.squeeze()
        sample2 = sample2.squeeze()

        ts_len = sample1.shape[1]
        lambada = torch.distributions.Uniform(0, 0.5).sample().item()
        sample1_size = int(ts_len * lambada)
        sample2_size = ts_len - sample1_size 

        chunk1 = sample1[:, :sample1_size]  
        chunk2 = sample2[:, sample1_size:]  
        result = torch.cat((chunk1, chunk2), dim=1)

        return result


class WISDM(Dataset):
    def __init__(self, data_path='data/200', is_test=False,transform=False):
        if is_test:
            #file_path = os.path.join(data_path, 'WISDM_normalize_test.pt')
            file_path = os.path.join(data_path, 'WISDM_nooverlap_test.pt')
        else:
            #file_path = os.path.join(data_path, 'WISDM_normalize_train.pt')
            file_path = os.path.join(data_path, 'WISDM_nooverlap_train.pt')

        # Load the data
        data = torch.load(file_path)

        # Ensure that data['X'] and data['y'] are numeric arrays
        if isinstance(data['X'], np.ndarray) and data['X'].dtype == np.object_:
            data['X'] = np.array(data['X'], dtype=np.float32)
        if isinstance(data['y'], np.ndarray) and data['y'].dtype == np.object_:
            data['y'] = np.array(data['y'], dtype=np.int64)

        self.X = torch.tensor(data['X'], dtype=torch.float32).permute(0, 2, 1).unsqueeze(1) # bs, 1, nvar, L
        self.y = torch.tensor(data['y'], dtype=torch.long)
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx]
        sample_y = self.y[idx]

        if self.transform:
            sample_X = sample_X.unsqueeze(0)
            sample_X = F.interpolate(sample_X, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0).repeat(1, 3, 1, 1).squeeze() 
        
        else:
            new_length = int(sample_X.shape[-1] * 50 / 20)  
            sample_X = F.interpolate(sample_X, size=new_length, mode='linear', align_corners=True)
        # #sample_X = sample_X.squeeze()

        return sample_X, sample_y

class IMWSHA(Dataset):
    def __init__(self, data_path='data/200', is_test=False,transform=False):
        if is_test:
            file_path = os.path.join(data_path, 'IMWSHA_nooverlap_test.pt')
        else:
            file_path = os.path.join(data_path, 'IMWSHA_nooverlap_train.pt')

        # Load the data
        data = torch.load(file_path)

        # Ensure that data['X'] and data['y'] are numeric arrays
        if isinstance(data['X'], np.ndarray) and data['X'].dtype == np.object_:
            data['X'] = np.array(data['X'], dtype=np.float32)
        if isinstance(data['y'], np.ndarray) and data['y'].dtype == np.object_:
            data['y'] = np.array(data['y'], dtype=np.int64)

        self.X = torch.tensor(data['X'], dtype=torch.float32).permute(0, 2, 1).unsqueeze(1) # bs, 1, nvar, L
        self.y = torch.tensor(data['y'], dtype=torch.long)
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx] # 1, 6, 400
        sample_y = self.y[idx]
        #print(sample_X.shape)
        
        if self.transform:
            # ViT Resample
            sample_X = sample_X.unsqueeze(0) 
            sample_X = F.interpolate(sample_X, size=(224, 224), 
                                     mode='bilinear', 
                                     align_corners=False).squeeze(0).repeat(1, 3, 1, 1).squeeze() 
        else:
            # Rsample to sampling rate 50Hz
            new_length = int(sample_X.shape[-1] * 50 / 100)  
            sample_X = F.interpolate(sample_X, size=new_length, mode='linear', align_corners=True)

        return sample_X, sample_y


class Oppo(Dataset):
    def __init__(self, data_path='data/200', is_test=False,transform=False):
        self.sample_rate = 30
        if is_test:
            x_path = os.path.join(data_path, 'oppo_30hz_w10','test_x.npy')
            y_path = os.path.join(data_path, 'oppo_30hz_w10','test_y.npy')
        else:
            x_path = os.path.join(data_path, 'oppo_30hz_w10','train_x.npy')
            y_path = os.path.join(data_path, 'oppo_30hz_w10','train_y.npy')

        x = np.load(x_path) # bs, 300, 3
        y = np.load(y_path) # list of labels
        y = y-1 # make labels start from 0

        self.X = torch.tensor(x, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1) # bs, 1, nvar, L
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx] # 1, 3, 300
        sample_y = self.y[idx]
        #print(sample_X.shape)
        
        if self.transform:
            # ViT Resample
            sample_X = sample_X.unsqueeze(0) 
            sample_X = F.interpolate(sample_X, size=(224, 224), 
                                     mode='bilinear', 
                                     align_corners=False).squeeze(0).repeat(1, 3, 1, 1).squeeze() 
        else:
            # Rsample to sampling rate 50Hz
            new_length = int(sample_X.shape[-1] * 50 / self.sample_rate)  
            sample_X = F.interpolate(sample_X, size=new_length, mode='linear', align_corners=True)

        return sample_X, sample_y
    
import pickle
class Capture24(Dataset):
    def __init__(self, data_path='data/200', nb_classes=4,
                 is_test=False, transform=False):
        self.sample_rate = 100
        self.transform = transform
        
        self.x_path = '/home/jovyan/persistent-data/leo/data/capture24/prepared_data/X.npy'
        self.x = np.load(self.x_path, mmap_mode='r')  # memory-mapped, not fully loaded
        
        if nb_classes == 4:
            y = np.load('/home/jovyan/persistent-data/leo/data/capture24/prepared_data/Y_Walmsley2020.npy')
        else:
            y = np.load('/home/jovyan/persistent-data/leo/data/capture24/prepared_data/Y_WillettsSpecific2018.npy')

        classes, y = np.unique(y, return_inverse=True)

        with open('/home/jovyan/persistent-data/leo/data/capture24/prepared_data/split.pkl', 'rb') as f:
            split = pickle.load(f)

        self.indices = np.where(split['test'] if is_test else split['train'])[0]
        self.y = y[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        sample_x = self.x[i]  # shape: (1000, 3)
        sample_y = self.y[idx]

        sample_x = torch.tensor(sample_x, dtype=torch.float32).permute(1, 0).unsqueeze(0)  # (1, 3, 1000)

        if self.transform:
            sample_x = sample_x.unsqueeze(0)
            sample_x = F.interpolate(sample_x, size=(224, 224), mode='bilinear', align_corners=False)
            sample_x = sample_x.squeeze(0).repeat(1, 3, 1, 1).squeeze()
        else:
            new_length = int(sample_x.shape[-1] * 50 / self.sample_rate)
            sample_x = F.interpolate(sample_x, size=new_length, mode='linear', align_corners=True)

        return sample_x, sample_y








if __name__ == "__main__":
    print("Starting dataset loading and testing...")

    # Parameters
    data_path = 'data/200'  # Change this to your actual data path
    batch_size = 4
    normalization = True  # Set this to True if you want to apply normalization
    nb_classes = 6

    # Verify data path
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist.")
    else:
        print(f"Using data path: {data_path}")

    # Create train and test datasets
    # train_dataset = UCIHAR(data_path=data_path, is_test=False, normalization=normalization,nb_classes=nb_classes)
    # test_dataset = UCIHAR(data_path=data_path, is_test=True, normalization=normalization,nb_classes=nb_classes)

    train_dataset = WISDM(data_path=data_path,is_test=False)
    test_dataset = WISDM(data_path=data_path,is_test=True)

    # Print the length of the datasets
    print(f"Train dataset length: {len(train_dataset)}")
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