from core.callback.base_callback import BaseCallBack
from core.callback.built_in import Backward,Forward,Plotter
from core.audit.metrics import Metrics
import torch.nn.functional as F
import torch
import numpy as np

import torch.nn as nn

class CBS(BaseCallBack):
    def __init__(self,early_stopping,patience):
        super().__init__(early_stopping,patience)

    def before_fit(self,*args, **kwargs):
        pass
    def before_epoch(self,*args,**kwargs):
        recorder = self.engine.recorder[self.engine.active_mode]
        recorder.__reset__()
    def before_batch(self,*args, **kwargs):
        pass
    def before_forward_step(self,*args,**kwargs):
        pass
    def after_forward_step(self,*args,**kwargs):
        pass
    def fetch_data(self,*args,**kwargs):
        super().fetch_data()
    def loss_function(self,*args,**kwargs):
        super().loss_function()
    def forward_step(self,*args,**kwargs):
        if self.engine.scaler is not None:
            outputs = Forward().fp16(model=self.engine.model,data=self.engine.data)
        else:
            outputs = Forward().standard(model=self.engine.model,data=self.engine.data)
        return outputs
    def backward_step(self,*args,**kwargs):
        if self.engine.scaler is not None:
            Backward().fp16(loss=self.engine.loss,optimizer=self.engine.optimizer,scaler=self.engine.scaler)
        else:
            Backward().standard(loss=self.engine.loss,optimizer=self.engine.optimizer)
    def after_batch(self,*args, **kwargs):
        recorder = self.engine.recorder[self.engine.active_mode]
        recorder.__update_batch__(self.engine.loss.item())
    def after_epoch(self, *args, **kwargs):
        recorder = self.engine.recorder[self.engine.active_mode]
        recorder.__update_epoch__()
    def after_train_eval(self,*args, **kwargs):
        recorder = self.engine.recorder
        print(f"Training F1 Score: "
              f"{recorder['train'].metrics['f1_score']}, "
              f"Validation F1 Score: "
              f"{recorder['eval'].metrics['f1_score']}")
        print(f"Training F1 Score: "
              f"{recorder['train'].metrics['f1_score']},"
              f" Validation F1 Score: "
              f"{recorder['eval'].metrics['f1_score']}")
    def after_fit(self,*args, **kwargs):
        Plotter().losses(self.engine.recorder['train'].loss_history,self.engine.recorder['eval'].loss_history)