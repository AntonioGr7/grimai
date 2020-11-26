from core.exception.CallBackException import CallBackBaseException

class CallBack:
    event_names = []

    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
    def __repr__(self):
        return type(self).__name__
    def __call__(self, event_name,**kwargs):
        if event_name in self.event_names:
            res = getattr(self, event_name)(**kwargs)
            return res
        else:
            raise CallBackBaseException()
