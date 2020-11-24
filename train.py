#### Example on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from engine.engine import Engine
from model.example_model import ExampleModel
from engine.engine import Engine
from cbs import CBS


root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
valid_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
valid_loader = torch.utils.data.DataLoader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=False)

custom_model = ExampleModel(input_dimension=28*28,hidden_dimension=500,output_dimension=10)
optimizer = optim.SGD(custom_model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
cbs = CBS()

device = [0]
engine = Engine(model=custom_model,optimizer=optimizer,cbs=cbs,fp16=True,scheduler=None,device=device)
engine.fit(epochs=10,train_dataloader=train_loader,valid_dataloader = valid_loader)