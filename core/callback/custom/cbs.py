from core.callback.base_callback import BaseCallBack
from core.callback.built_in import Backward,Forward,Plotter,EarlyStopping
from core.audit.metrics import Metrics
import torch.nn.functional as F
import numpy as np

import torch.nn as nn

class CBS(BaseCallBack):
    def __init__(self):
        super().__init__()
        self.early_stopping = EarlyStopping(patience=3,path="./",model_name="prova")

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
        return self.engine.batch[0].to(self.engine.device),self.engine.batch[1].to(self.engine.device)
    def loss_function(self,*args,**kwargs):
        outputs = self.engine.outputs
        targets = self.engine.targets
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(outputs,targets)
    def forward_step(self,*args,**kwargs):
        if self.engine.scaler is not None:
            outputs = Forward().fp16(model=self.engine.model,data=self.engine.data)
        else:
            outputs = Forward().standard(model=self.engine.model,data=self.engine.data)
        return outputs
    def backward_step(self,*args,**kwargs):
        loss = self.engine.loss
        if self.engine.scaler is not None:
            Backward().fp16(loss=self.engine.loss,optimizer=self.engine.optimizer,scaler=self.engine.scaler)
        else:
            Backward().standard(loss=self.engine.loss,optimizer=self.engine.optimizer)
        return loss
    def after_batch(self,*args, **kwargs):
        recorder = self.engine.recorder[self.engine.active_mode]
        recorder.__update_batch__(self.engine.loss.item())
        predictions = F.log_softmax(self.engine.outputs,dim=1)
        predictions = np.argmax(predictions.detach().cpu().numpy(),axis=1)
        targets = self.engine.targets.detach().cpu().numpy()
        metrics = Metrics(targets,predictions)
        recorder.metrics['accuracy'] = metrics.accuracy()
        recorder.metrics['f1_score'] = metrics.f1_score()
    def after_epoch(self, *args, **kwargs):
        recorder = self.engine.recorder[self.engine.active_mode]
        recorder.__update_epoch__()
        stop = self.early_stopping(self.engine.loss)
        if stop:
            exit("Early Stop. Model Saved")

    def after_fit(self,*args, **kwargs):
        Plotter().losses(self.engine.recorder['train'].loss_history,self.engine.recorder['eval'].loss_history)