import torch
from torch.utils.data import Dataset

class AddDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        cur = self.data[i]
        tokens = self.tokenizer.encode(cur)
        src = tokens[:-1]
        target = tokens[1:]
        
        return torch.tensor(src), torch.tensor(target)

