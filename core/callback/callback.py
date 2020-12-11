from core.exception.callback_exception import CallBackBaseException
class CallBack:
    event_names = ["before_fit","before_epoch","fetch_data","before_batch","before_forward_step","forward_step","after_forward_step",
                   "backward_step","loss_function","after_batch","after_epoch","after_train_eval","after_fit"]

    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
    def __repr__(self):
        return type(self).__name__
    def __call__(self, event_name,*args,**kwargs):
        if event_name in self.event_names:
            self.__dict__.update(kwargs)
            res = getattr(self, event_name)(**kwargs)
            return res
        else:
            raise CallBackBaseException()
