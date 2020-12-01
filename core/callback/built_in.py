from torch.cuda import amp
from datetime import datetime
import numpy as np
import os


class Forward():
    @staticmethod
    def standard(model,data):
        outputs = model(data)
        return outputs
    @staticmethod
    def fp16(model,data):
        with amp.autocast():
            outputs = model(data)
        return outputs


class Backward():
    @staticmethod
    def standard(loss,optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    @staticmethod
    def fp16(loss,optimizer,scaler):
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

class EarlyStopping():
    def __init__(self,patience=3,path="./",model_name="check_point"):
        self.path = path
        self.model_name = model_name
        self.best_loss = np.inf
        self.patience = 0
        self.max_patience = patience
    def __call__(self,loss):
        if loss <= self.best_loss:
            self.best_loss = loss
            self.patience = 0
            return False
        else:
            self.patience +=1
            if self.patience == self.max_patience:
                Saver().save(self.path,self.model_name + str(datetime.now()))
                return True
            return False


class Saver():
    @staticmethod
    def save(path,name):
        path = os.path.join(path,name)
        print(f"I'm calling Save on path:{path}")

from matplotlib import pyplot as plt

class Plotter():
    @staticmethod
    def losses(train_losses, valid_losses):
        plt.plot(range(len(train_losses)), train_losses, 'g', label='Training loss')
        plt.plot(range(len(valid_losses)), valid_losses, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    @staticmethod
    def plot(x,y,x_label,y_label,title=None):
        plt.plot(x, y, 'r', label="loss")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()



