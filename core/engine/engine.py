from core.engine.base_engine import BaseEngine


class Engine(BaseEngine):
    def __init__(self,model,optimizer,cbs,scheduler=None,fp16=None,parallelize=False,device=None,**kwargs):
        super().__init__(model,optimizer,cbs=cbs,scheduler=scheduler,fp16=fp16,parallelize=parallelize,device=device,**kwargs)
    def fit(self,epochs,train_dataloader,valid_dataloader):
        self.run_cbs(self.cbs.before_fit, **{"engine": self})
        for epoch in range(epochs):
            train_loss = self.train(train_dataloader,cbs=self.cbs)
            valid_loss = self.eval(valid_dataloader,cbs=self.cbs)
            print(f"Epoch:{epoch}, Training Loss:{train_loss}, Validation Loss:{valid_loss}")
            print(f"Training Accuracy: {self.recorder['train'].metrics['accuracy']}, Validation Accuracy: {self.recorder['eval'].metrics['accuracy']}")
            print(f"Training F1 Score: {self.recorder['train'].metrics['f1_score']}, Validation F1 Score: {self.recorder['eval'].metrics['f1_score']}")
        self.run_cbs(self.cbs.after_fit, **{"engine": self})


