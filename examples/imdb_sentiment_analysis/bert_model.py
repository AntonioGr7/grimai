import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout1 = nn.Dropout(0.8)
        self.fc1 = nn.Linear(768,50)
        self.dropout2 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(50,1)
    def forward(self,data):
        x = self.bert(input_ids=data[0],attention_mask=data[1])
        x = self.dropout1(x[0][:,0,:])
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
