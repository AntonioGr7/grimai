from callback.base_callback import BaseCallBack
from audit.plotter import Plotter
from audit.metrics import Metrics
import torch.nn as nn
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
        engine = kwargs['engine']
        recorder = engine.recorder[engine.active_mode]
        recorder.__reset__()
    def before_batch(self,*args, **kwargs):
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
    def after_batch(self,*args, **kwargs):
        engine = kwargs['engine']
        recorder = engine.recorder[engine.active_mode]
        recorder.__update_batch__(engine.loss.item())
        predictions = F.log_softmax(engine.outputs,dim=1)
        predictions = np.argmax(predictions.detach().cpu().numpy(),axis=1)
        targets = engine.targets.detach().cpu().numpy()
        metrics = Metrics(targets,predictions)
        recorder.metrics['accuracy'] = metrics.accuracy()
        recorder.metrics['f1_score'] = metrics.f1_score()
    def after_epoch(self, *args, **kwargs):
        engine = kwargs['engine']
        recorder = engine.recorder[engine.active_mode]
        recorder.__update_epoch__()
    def after_fit(self,*args, **kwargs):
        engine = kwargs['engine']
        plotter = Plotter()
        plotter.plot_losses(engine.recorder['train'].loss_history,engine.recorder['eval'].loss_history)

