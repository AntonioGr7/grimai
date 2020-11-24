from torch.cuda import amp
import functools
from tqdm import tqdm
from audit.recorder import Recorder
from audit.metrics import Metrics
import torch

class BaseEngine():
    def __init__(self,model,optimizer,cbs,scaler=None,scheduler=None,device=torch.device("cuda:0"),**kwargs):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.cbs = cbs
        self.active_mode = "train"
        self.recorder = {"train":Recorder(),"eval":Recorder()}
        self.__dict__.update(kwargs)

    def train(self,dataloader,cbs):
        self.active_mode = "train"
        self.model.train()
        self.run_cbs(cbs.before_epoch,**{"engine":self})
        tq = tqdm(dataloader, total=len(dataloader))
        self.batch_size = len(dataloader)
        for self.batch_index, self.batch in enumerate(tq):
            self.run_cbs(cbs.before_batch, **{"engine": self})
            self.run_cbs(cbs.before_forward_step,**{"engine":self})
            self.data,self.targets = self.run_cbs(cbs.fetch_data,**{"engine":self})
            self.outputs =  self.run_cbs(cbs.forward_step,**{"engine":self})
            self.run_cbs(cbs.after_forward_step)
            self.loss = self.run_cbs(cbs.loss_function,**{"engine":self})
            self.loss = self.run_cbs(cbs.backword_step, **{"engine": self})
            self.run_cbs(cbs.after_batch, **{"engine": self})
        self.run_cbs(cbs.after_epoch,**{"engine":self})
        return self.recorder[self.active_mode].loss


    def eval(self,dataloader,cbs):
        self.active_mode = "eval"
        self.model.eval()
        self.run_cbs(cbs.before_epoch, **{"engine": self})
        tq = tqdm(dataloader, total=len(dataloader))
        self.batch_size = len(dataloader)
        for self.batch_index, self.batch in enumerate(tq):
            with torch.no_grad():
                self.run_cbs(cbs.before_forward_step, **{"engine": self})
                self.data, self.targets = self.run_cbs(cbs.fetch_data, **{"engine": self})
                self.outputs = self.run_cbs(cbs.forward_step, **{"engine": self})
                self.run_cbs(cbs.after_forward_step)
                self.loss = self.run_cbs(cbs.loss_function, **{"engine": self})
                self.run_cbs(cbs.after_batch, **{"engine": self})
        self.run_cbs(cbs.after_epoch, **{"engine": self})
        return self.recorder[self.active_mode].loss

    def run_cbs(self,function,*args,**kwargs):
        @functools.wraps(function)
        def run(*args, **kwargs):
            return function(*args, **kwargs)
        return run(*args,**kwargs)