from core.callback.base_callback import BaseCallBack
from core.audit.plotter import Plotter
from core.audit.metrics import Metrics
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np

import torch.nn as nn

class CBS(BaseCallBack):
    def __init__(self):
        super().__init__()

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
            with amp.autocast():
                outputs = self.engine.model(self.engine.data)
        else:
            outputs = self.engine.model(self.engine.data)
        return outputs
    def backword_step(self,*args,**kwargs):
        loss = self.engine.loss
        self.engine.optimizer.zero_grad()
        if self.engine.scaler is not None:
            self.engine.scaler.scale(loss).backward()
            self.engine.scaler.step(self.engine.optimizer)
            self.engine.scaler.update()
        else:
            loss.backward()
            self.engine.optimizer.step()
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
    def after_fit(self,*args, **kwargs):
        plotter = Plotter()
        plotter.plot_losses(self.engine.recorder['train'].loss_history,self.engine.recorder['eval'].loss_history)
