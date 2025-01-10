import torch
import torch.utils
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    ''' Dataset class '''
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]