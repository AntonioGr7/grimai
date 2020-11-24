from callback.callback import CallBack
from exception import CallBackException
class BaseCallBack(CallBack):
    def __init__(self):
        super().__init__()
    def before_epoch(self, *args, **kwargs):
        pass
    def after_epoch(self, *args, **kwargs):
        pass
    def before_forward_step(self, *args, **kwargs):
        pass
    def after_forward_step(self, *args, **kwargs):
        pass
    def fetch_data(self,*args,**kwargs):
        raise CallBackException("fetch data must be implemented")
    def loss_function(self,*args,**kwargs):
        raise CallBackException("loss function must be implemented")
    def forward_step(self,*args,**kwargs):
        raise CallBackException("forward step must be implemented")
    def backword_step(self,*args,**kwargs):
        raise CallBackException("backword step data must be implemented")