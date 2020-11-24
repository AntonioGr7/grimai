from torch.cuda import amp
import functools
from tqdm import tqdm
import torch

class BaseEngine():
    def __init__(self,model,optimizer,cbs,scaler=None,scheduler=None,device=torch.device("cuda:0"),**kwargs):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.cbs = cbs
        self.__dict__.update(kwargs)

    def train(self,dataloader,cbs):
        self.model.train()
        self.run_cbs(cbs.before_epoch,**{"engine":self})
        tq = tqdm(dataloader, total=len(dataloader))
        for batch_index, batch in enumerate(tq):
            self.batch_index = batch_index
            self.batch = batch
            self.run_cbs(cbs.before_forward_step,**{"engine":self})
            self.data,self.targets = self.run_cbs(cbs.fetch_data,**{"engine":self})
            self.outputs =  self.run_cbs(cbs.forward_step,**{"engine":self})
            self.run_cbs(cbs.after_forward_step)
            self.loss = self.run_cbs(cbs.loss_function,**{"engine":self})
            self.loss = self.run_cbs(cbs.backword_step, **{"engine": self})
        self.run_cbs(cbs.after_epoch,**{"engine":self})
        return self.loss.item()


    def eval(self,dataloader,cbs):
        self.model.eval()
        self.run_cbs(cbs.before_epoch, **{"engine": self})
        tq = tqdm(dataloader, total=len(dataloader))
        for batch_index, batch in enumerate(tq):
            with torch.no_grad():
                self.batch_index = batch_index
                self.batch = batch
                self.run_cbs(cbs.before_forward_step, **{"engine": self})
                self.data, self.targets = self.run_cbs(cbs.fetch_data, **{"engine": self})
                self.outputs = self.run_cbs(cbs.forward_step, **{"engine": self})
                self.run_cbs(cbs.after_forward_step)
                self.loss = self.run_cbs(cbs.loss_function, **{"engine": self})
        self.run_cbs(cbs.after_epoch, **{"engine": self})
        return self.loss.item()

    def run_cbs(self,function,*args,**kwargs):
        @functools.wraps(function)
        def run(*args, **kwargs):
            return function(*args, **kwargs)
        return run(*args,**kwargs)