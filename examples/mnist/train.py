#### Example on mnist_
import os
import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from examples.mnist.mnist_model import MNISTModel
from engine.engine import Engine
from callback.custom.cbs import CBS


if __name__ == "__main__":

    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    valid_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

    BATCH_SIZE = 64
    EPOCHS = 10

    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=BATCH_SIZE,
                     shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
                    dataset=valid_set,
                    batch_size=BATCH_SIZE,
                    shuffle=False)

    my_model = MNISTModel(input_dimension=28*28,hidden_dimension=500,output_dimension=10)
    optimizer = optim.SGD(my_model.parameters(), lr=0.001, momentum=0.9)
    cbs = CBS()

    device = [0]
    engine = Engine(model=my_model,optimizer=optimizer,cbs=cbs,fp16=True,scheduler=None,device=device)
    engine.fit(epochs=10,train_dataloader=train_loader,valid_dataloader = valid_loader)