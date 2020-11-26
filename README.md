# GrimAI
GrimAI is a general purpose library build on top of pytorch. 

The objective of the library is to provide a simple and flexible approach to model training. 

# How to use

  - Import your data and create your dataloader
  - Implement your Callback class that inherits from BaseCallBack. You need to implement the callback function that you want to use during the training. Each of these callbacks will be invoked automatically at each of the steps described by the method name. 
```python
    from core.callback.base_callback import BaseCallBack
    class MyCBS(BaseCallBack):
    def __init__(self):
        super().__init__()
    def before_fit(self,*args, **kwargs):pass
    def before_epoch(self,*args,**kwargs):pass
    def before_batch(self,*args, **kwargs):pass
    def before_forward_step(self,*args,**kwargs):pass
    def after_forward_step(self,*args,**kwargs):pass
    def fetch_data(self,*args,**kwargs):print("Mandatory")
    def loss_function(self,*args,**kwargs):print("Mandatory")
    def forward_step(self,*args,**kwargs):print("Mandatory")
    def backword_step(self,*args,**kwargs):print("Mandatory")
    def after_batch(self,*args, **kwargs):pass
    def after_epoch(self, *args, **kwargs):pass
    def after_fit(self,*args, **kwargs):pass
```
  - Within the inherited class you will always have access to the engine, containing all the variables and methods you need. For example:
```python
    def fetch_data(self,*args,**kwargs):
        return self.engine.batch[0].to(self.engine.device),self.engine.batch[1].to(self.engine.device)
    def forward_step(self,*args,**kwargs):
        if self.engine.scaler is not None:
            with amp.autocast():
                outputs = self.engine.model(self.engine.data)
        else:
            outputs = self.engine.model(self.engine.data)
        return outputs
    def backword_step(self,*args,**kwargs):
        loss = self.engine.loss
        self.engine.optimizer.zero_grad()
        if self.engine.scaler is not None:
            self.engine.scaler.scale(loss).backward()
            self.engine.scaler.step(self.engine.optimizer)
            self.engine.scaler.update()
        else:
            loss.backward()
            self.engine.optimizer.step()
        return loss
```
 - Pass your callback in the invocation method:
```python
    optimizer = optim.SGD(my_model.parameters(), lr=0.001, momentum=0.9)
    cbs = CBS()
    device = [0]
    engine = Engine(model=my_model,optimizer=optimizer,cbs=cbs,fp16=True,scheduler=None,device=device)
    engine.fit(epochs=10,train_dataloader=train_loader,valid_dataloader = valid_loader)
```

See the MNIST example for details.  

### Features
If you use the CBS already provided:
 - Mixed Precision Training already available passing fp16=True in the engine
 - Parallel training on GPUs available passing an array to device. For example with [0,1] your model will use GPU:0 and GPU:1

You can use this CallBack class and inject your special function. For example:
```python
    cbs = CBS()
    def fetch_data(*args,**kwargs):
        print("my fetch data")
    cbs.fetch_data = fetch_data
    device = [0,1]
    engine = Engine(model=my_model,optimizer=optimizer,cbs=cbs,fp16=True,scheduler=None,device=device)
    engine.fit(epochs=10,train_dataloader=train_loader,valid_dataloader = valid_loader)
```
### What's next
 - More stable callbacks class and function available by default
 - Create dataloader automatically for some class of data
 - More examples
 - Export your model with ONNX
 - Installing from pip

### Installation

### Developer informations
[Linkedin](https://www.linkedin.com/in/antonio-grimaldi-99489a122/)

License
----
MIT
