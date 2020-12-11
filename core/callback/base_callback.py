from core.callback.callback import CallBack
from core.exception.callback_exception import CallBackBaseException
from core.callback.built_in import EarlyStopping
from time import time

class BaseCallBack(CallBack):
    def __init__(self,early_stopping=False,patience=3):
        super().__init__()
        self.do_early_stopping = early_stopping
        self.patience = patience
        self.model_output_name = "model"+str(time()) + ".bin"
        self.early_stopping = EarlyStopping(patience=patience, path="./", model_name=self.model_output_name)
        '''
        When you inherit from this class, you must implement all the methods you intend to use 
        during your training cycle. 
        For example "before_forward_step" will be called exactly before the forward step 
        (both in the train and in the eval). 
        In each inherited callback class you will always have an engine object available, 
        accessible with self.engine. 
        This object contains all the variables you will need through each step. 
        self.engine:
            Immediatly available:
                - device -> on which device (cuda:0,cuda:1)
                - fp16 -> boolean if you are training on mixed precision
                - parallelize -> boolean if you are training on parallel gpu
                - model -> model to train
                - optimizer 
                - scheduler 
                - active_mode -> "train","eval" or "find_learning_rate"
                - recorder -> {"train":Recorder(),"eval":Recorder(),"find_learning_rate":Recorder()}
                           To save loss and other metric on each step
                - train_dataloader
                - valid_dataloader
                - scaler -> For mixed precision training if not None
            After before_epoch:
                - batch_index -> index of active batch
                - batch -> batch of data
            After fetch_data:
                - data -> Data in the format you choose in your fetch_data callback
                - targets -> Target in the format you choose in your fetch_data callback
            After loss_function:
                - loss -> loss calculated with your custom loss_function callback
            After forward_step:
                - outputs -> outputs from your forward step
        '''

    def before_fit(self,*args, **kwargs):
        pass
    def before_epoch(self, *args, **kwargs):
        pass
    def before_batch(self,*args, **kwargs):
        pass
    def fetch_data(self,*args,**kwargs):
        raise CallBackException("fetch data must be implemented")
    def before_forward_step(self, *args, **kwargs):
        pass
    def forward_step(self, *args, **kwargs):
        raise CallBackException("forward step must be implemented")
    def after_forward_step(self, *args, **kwargs):
        pass
    def backward_step(self, *args, **kwargs):
        raise CallBackException("backword step data must be implemented")
    def loss_function(self,*args,**kwargs):
        raise CallBackException("loss function must be implemented")
    def after_batch(self,*args, **kwargs):
        pass
    def after_epoch(self, *args, **kwargs):
        pass
    def early_stopping_step(self,*args, **kwargs):
        if self.do_early_stopping:
            stop = self.early_stopping(self.engine.loss)
            if stop:
                exit("Early Stop. Model Saved")
    def after_train_eval(self,*args,**kwargs):
        pass
    def after_fit(self,*args, **kwargs):
        pass

