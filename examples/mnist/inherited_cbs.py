import torch.nn as nn
import torch
from core.callback.built_in import Plotter
from core.callback.custom.cbs import CBS
import torch.nn.functional as F
from core.audit.metrics import Metrics
import numpy as np


class InheritedCBS(CBS):
    def __init__(self,early_stopping=False,patience=3):
        super().__init__(early_stopping,patience)
    def fetch_data(self,*args,**kwargs):
        return self.engine.batch[0].to(self.engine.device),self.engine.batch[1].to(self.engine.device)
    def loss_function(self,*args,**kwargs):
        outputs = self.engine.outputs
        targets = self.engine.targets
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(outputs, targets)
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