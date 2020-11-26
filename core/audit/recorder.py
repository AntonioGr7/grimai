from matplotlib import pyplot as plt
import numpy as np

class Recorder:
    def __init__(self):
        self.loss = 0
        self.loss_history = []
        self.best_loss = np.inf
        self.metrics = {}

    def __reset__(self):
        self.count = 0
        self.loss = 0
        self.accuracy = 0
        self.f1_score = 0

    def __update_batch__(self,loss):
        self.loss += loss
        self.count +=1

    def __update_epoch__(self):
        self.loss = self.loss / self.count
        self.loss_history.append(self.loss)
