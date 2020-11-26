from torch.cuda import amp
import functools
from tqdm import tqdm
from core.audit.recorder import Recorder
import torch.nn as nn
import torch

class BaseEngine():
    def __init__(self,model,optimizer,cbs,scheduler=None,fp16=False,parallelize=False,device=torch.device("cuda:0"),**kwargs):
        self.device = device
        self.fp16 = fp16
        self.parallelize = parallelize
        self.configure()
        self.model = model.to(self.device)
        self.optimizer = optimizer
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
            self.outputs = self.run_cbs(cbs.forward_step,**{"engine":self})
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

    def configure(self):
        if isinstance(self.device, list):
            if self.parallelize:
                if torch.cuda.device_count() > 1:
                    self.model = nn.DataParallel(self.model, device_ids=self.device)
                    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    print(f"Using Data Parallel on GPUs:{str(self.device)}")
                else:
                    raise Exception("If you set 'parallelize'=true you need to indicate devices as list of integer")
            else:
                print(f"Using single GPU:{str(self.device)}")
                self.device = torch.device(f"cuda:{self.device[0]}" if torch.cuda.is_available() else "cpu")
        if self.fp16:
            self.scaler = amp.GradScaler()
            print("Using FP16 Mixed Precision Training")
        else:
            self.scaler = None