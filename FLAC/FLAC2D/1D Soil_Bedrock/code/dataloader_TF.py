import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

class TTF_dataset(Dataset):
    def __init__(self, TTF_values, Vs_values):
        self.TTF_values = TTF_values
        self.Vs_values = Vs_values
        
    def __len__(self):
        return len(self.TTF_values)
    
    def __getitem__(self, idx):
        # Target data
        target_data = self.TTF_values[idx]
        target_data = torch.tensor(target_data, dtype=torch.float32).unsqueeze(1)
        # Input data
        input_data = np.nan_to_num(self.Vs_values[idx], nan=0.0, posinf=0.0, neginf=0.0)
        input_data = torch.tensor(input_data, dtype=torch.float32).repeat(target_data.shape[0], 1)

        return input_data, target_data


def get_dataloader(TTF_values, Vs_values, batch_size=50, training_fraction=0.8):
    dataset = TTF_dataset(TTF_values, Vs_values)
    train_size = int(0.7 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

def get_data(TF_path, Vs_path, freq_path):
    # Frequency data
    freq_data = np.genfromtxt(freq_path, delimiter=",") 
    # Target data
    with open(TF_path, "rb") as f:
        TTF_data = pickle.load(f)

    # Input data
    with open(Vs_path, "rb") as f:
        Vs_data = pickle.load(f)
    
    # Convert to numpy arrays
    TTF_data = np.array(TTF_data)
    Vs_data = np.array(Vs_data)

    return TTF_data, Vs_data, freq_data