
class Metric():
    def __init__(self):
        self.losses = []
        self.loss = 0
        self.count = 0
    
    def update(self,loss,batch_size=1):
        self.loss += loss
        self.count += batch_size
    
    def avg_loss(self):
        return self.loss / self.count