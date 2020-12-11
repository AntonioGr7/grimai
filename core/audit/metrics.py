from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class Metrics():
    def __init__(self,predictions,targets):
        self.predictions = predictions
        self.targets = targets
    def accuracy(self):
        return accuracy_score(self.targets, self.predictions)
    def f1_score(self):
        return f1_score(self.targets, self.predictions, average='micro')