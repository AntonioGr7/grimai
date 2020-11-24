import torch.nn as nn
import torch.nn.functional as F

class ExampleModel(nn.Module):
    def __init__(self,input_dimension,hidden_dimension,output_dimension):
        super().__init__()
        self.l1 = nn.Linear(input_dimension,hidden_dimension)
        self.l2 = nn.Linear(hidden_dimension,output_dimension)
    def forward(self,x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x