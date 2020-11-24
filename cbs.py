from callback.base_callback import BaseCallBack
from torch.cuda import amp
import torch.nn as nn

class CBS(BaseCallBack):
    def __init__(self):
        super().__init__()

    def before_epoch(self,*args,**kwargs):
        pass
    def after_epoch(self,*args,**kwargs):
        pass
    def before_forward_step(self,*args,**kwargs):
        pass
    def after_forward_step(self,*args,**kwargs):
        pass
    def fetch_data(self,*args,**kwargs):
        engine = kwargs['engine']
        return engine.batch[0].to(engine.device),engine.batch[1].to(engine.device)
    def loss_function(self,*args,**kwargs):
        outputs = kwargs['engine'].outputs
        targets = kwargs['engine'].targets
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(outputs,targets)
    def forward_step(self,*args,**kwargs):
        engine = kwargs['engine']
        if engine.scaler is not None:
            with amp.autocast():
                outputs = engine.model(engine.data)
        else:
            outputs = engine.model(engine.data)
        return outputs
    def backword_step(self,*args,**kwargs):
        engine = kwargs['engine']
        loss = engine.loss
        engine.optimizer.zero_grad()
        if engine.scaler is not None:
            engine.scaler.scale(loss).backward()
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
        else:
            loss.backward()
            engine.optimizer.step()
        return loss

