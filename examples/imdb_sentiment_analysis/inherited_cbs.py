import torch.nn as nn
import torch
from core.callback.custom.cbs import CBS
from core.audit.metrics import Metrics
import numpy as np


class InheritedCBS(CBS):
    def __init__(self,early_stopping=False,patience=3):
        super().__init__(early_stopping,patience)
    def fetch_data(self,*args,**kwargs):
        data = (self.engine.batch['ids'].to(self.engine.device),
                self.engine.batch['mask'].to(self.engine.device))
        target = self.engine.batch['target'].to(self.engine.device)
        return data,target
    def loss_function(self,*args,**kwargs):
        outputs = self.engine.outputs
        targets = self.engine.targets
        targets = targets.unsqueeze(dim=1)
        if self.engine.scaler is not None:
            targets = targets.type(torch.float16)
        loss_fct = nn.BCEWithLogitsLoss()
        return loss_fct(outputs,targets)
    def after_batch(self, *args, **kwargs):
        recorder = self.engine.recorder[self.engine.active_mode]
        recorder.__update_batch__(self.engine.loss.item())
        outputs = np.array(self.engine.outputs.detach().cpu()) >= 0.5
        targets = self.engine.targets.detach().cpu().numpy()
        metrics = Metrics(targets, outputs)
        recorder.accuracy += metrics.accuracy()
        recorder.f1_score += metrics.f1_score()
    def after_epoch(self, *args, **kwargs):
        recorder = self.engine.recorder[self.engine.active_mode]
        recorder.metrics['accuracy'] = recorder.accuracy / recorder.count
        recorder.metrics['f1_score'] = recorder.f1_score / recorder.count
        recorder.__update_epoch__()
        stop = self.early_stopping(self.engine.loss)
        if stop:
            exit("Early Stop. Model Saved")