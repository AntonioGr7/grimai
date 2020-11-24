from engine.base_engine import BaseEngine
from tqdm import tqdm
import torch
import torch.nn as nn
from cbs import CBS
from torch.cuda import amp

class Engine(BaseEngine):
    def __init__(self,model,optimizer,cbs,fp16=None,scheduler=None,device=None,**kwargs):
        if isinstance(device, list):
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=device)
                print(f"Using Data Parallel on GPUs:{str(device)}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if fp16:
            scaler = amp.GradScaler()
            print("Using FP16 Mixed Precision Training")
        super().__init__(model,optimizer,cbs=cbs,scaler=scaler,scheduler=None,device=device,**kwargs)
    def fit(self,epochs,train_dataloader,valid_dataloader):
        for epoch in range(epochs):
            train_loss = self.train(train_dataloader,cbs=self.cbs)
            valid_loss = self.eval(valid_dataloader,cbs=self.cbs)
            print(f"Epoch:{epoch}, Training Loss:{train_loss}, Validation Loss:{valid_loss}")

