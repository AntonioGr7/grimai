from engine.base_engine import BaseEngine
import torch
import torch.nn as nn
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
        self.run_cbs(self.cbs.before_fit, **{"engine": self})
        for epoch in range(epochs):
            train_loss = self.train(train_dataloader,cbs=self.cbs)
            valid_loss = self.eval(valid_dataloader,cbs=self.cbs)
            print(f"Epoch:{epoch}, Training Loss:{train_loss}, Validation Loss:{valid_loss}")
            print(f"Training Accuracy: {self.recorder['train'].metrics['accuracy']}, Validation Accuracy: {self.recorder['eval'].metrics['accuracy']}")
            print(f"Training F1 Score: {self.recorder['train'].metrics['f1_score']}, Validation F1 Score: {self.recorder['eval'].metrics['f1_score']}")
        self.run_cbs(self.cbs.after_fit, **{"engine": self})

