import torch

class Dataset():
    def __init__(self,features,targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self,index):
        return {
            "x":torch.tensor(self.features[index,:],dtype=torch.float),
            "y":torch.tensor(self.targets[index,:],dtype=torch.float)
        }