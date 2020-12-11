from core.engine.base_engine import BaseEngine
from core.callback.built_in import Plotter
import copy


class Engine(BaseEngine):
    def __init__(self,model,optimizer,cbs,scheduler=None,fp16=None,parallelize=False,device=None,**kwargs):
        super().__init__(model,optimizer,cbs=cbs,scheduler=scheduler,fp16=fp16,parallelize=parallelize,device=device,**kwargs)
    def fit(self,epochs,train_dataloader,valid_dataloader):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.run_cbs(self.cbs.before_fit, **{"engine": self})
        for epoch in range(epochs):
            train_loss = self.train(train_dataloader,cbs=self.cbs)
            valid_loss = self.eval(valid_dataloader,cbs=self.cbs)
            print(f"Epoch:{epoch}, Training Loss:{train_loss}, Validation Loss:{valid_loss}")
            self.run_cbs(self.cbs.after_train_eval,**{"engine":self})
        self.run_cbs(self.cbs.after_fit, **{"engine": self})

class LRFinder(BaseEngine):
    def __init__(self,model,optimizer,cbs,scheduler=None,fp16=None,parallelize=False,device=None,**kwargs):
        super().__init__(model,optimizer,cbs=cbs,scheduler=scheduler,fp16=fp16,parallelize=parallelize,device=device,**kwargs)

    def find_lr(self, dataloader, init_value=1e-7, final_value=100000., beta=0.98):
        initial_model = copy.deepcopy(self.model)
        initial_optimizer = type(self.optimizer)(initial_model.parameters(), lr=self.optimizer.defaults['lr'])
        initial_optimizer.load_state_dict(self.optimizer.state_dict())
        log_lrs, losses = self.find_learning_rate(dataloader, self.cbs, init_value, final_value, beta)
        Plotter().plot(x=log_lrs, y=losses, x_label="learning rate (log scale)", y_label="loss",
                       title="Learning rate finder")
        self.model = initial_model
        self.optimizer.load_state_dict(initial_optimizer.state_dict())