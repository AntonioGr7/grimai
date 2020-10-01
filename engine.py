import torch.nn as nn
import torch
import metric as Metric

class Engine:
    def __init__(self,model,optimizer,scheduler=None,device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self,data_loader):
        self.model.train()
        metric = Metric.Metric()
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            outputs = self.model(inputs)
            loss = self.model.loss_fn(targets,outputs)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            metric.update(loss.item(),len(data))
        return metric.avg_loss()
        
    def evaluate(self,data_loader):
        self.model.eval()
        metric = Metric.Metric()
        for data in data_loader:
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.model.loss_fn(targets,outputs)
            metric.update(loss.item(),len(data))
        return metric.avg_loss()

    def predict(self,data_loader):
        self.model.eval()
        predictions_output = []
        with torch.no_grad():
            for data in data_loader:
                inputs = data['x'].to(self.device)
                predictions = self.model(inputs)
                predictions = predictions.cpu()
                predictions_output.append(predictions)
        return predictions_output
