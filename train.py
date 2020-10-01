import torch
import torch.utils.data as torchData
import pandas as pd
import data
import utils
import engine
import model
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Train deep learning model')
parser.add_argument("--lr", "-lr",nargs='?',help="set learning rate",const=0.001)
parser.add_argument("--device", "-device",nargs='?',help="set device [cuda,cpu]",const="cpu")
parser.add_argument("--epochs","-epochs",nargs='?',help="set number of epochs",const=100)
parser.add_argument("--seed","-seed",nargs='?',help="set seed in the shuffling",const=None)
args = parser.parse_args()


if __name__ == "__main__":

    DEVICE = args.device
    EPOCHS = int(args.epochs)
    LEARNING_RATE = float(args.lr)
    NUM_WORKERS = 8
    BATCH_SIZE = 64
    EARLY_STOPPING = 10
    if args.seed is None:
        SEED = None
    else:
        SEED = int(args.seed)

    print(f"Start training on device: {DEVICE}, learning rate {LEARNING_RATE}, number of epochs: {EPOCHS}")
    
    ''' Read Data and create data loader '''

    train,valid,testing = utils.Utils.read_data() 
    train_dataset = data.Dataset(features=train[0],targets=train[1])
    valid_dataset = data.Dataset(features=valid[0],targets=valid[1])
    testing_dataset = data.Dataset(features=testing[0],targets=testing[1])

   

    train_dataloader = torchData.DataLoader(
        dataset=train_dataset,batch_size = BATCH_SIZE,num_workers=NUM_WORKERS
    )
    valid_dataloader = torchData.DataLoader(
        dataset=valid_dataset,batch_size = BATCH_SIZE,num_workers=NUM_WORKERS
    )
    testing_dataloader = torchData.DataLoader(
        dataset=testing_dataset,batch_size= BATCH_SIZE,num_workers=NUM_WORKERS
    )
    ########################################################

    ''' 
    Define model in the model.py file and init it
    Define optimizer
    Define scheduler (if you want one)
    Pass all to the engine
    '''
    model = model.Model()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        patience=3,
        optimizer=optimizer,
        mode="min"
    )
    engine = engine.Engine(model,optimizer,scheduler,DEVICE)

    best_loss = np.inf
    early_stopping_count = 0

    validation_losses = []
    training_losses = []
    history_epochs = []

    for epoch in range(EPOCHS):
        train_loss = engine.train(train_dataloader)
        validation_loss = engine.evaluate(valid_dataloader)

        training_losses.append(train_loss)
        validation_losses.append(validation_loss)
        history_epochs.append(epoch)
        print("Epoch, Training loss, Validation loss: {}, {}, {}".format(str(epoch),str(train_loss),str(validation_loss)))
        if validation_loss < best_loss:
            early_stopping_count =0
            best_loss = validation_loss
        else:
            early_stopping_count +=1
        if early_stopping_count > EARLY_STOPPING:
            torch.save(model.state_dict(),f"model_output/model_{epoch}.bin")
            break
    if early_stopping_count <= EARLY_STOPPING:
        torch.save(model.state_dict(),f"model_output/model_final.bin")
    utils.Utils.save_losses_graph(training_losses,training_losses,path="model_output/losses.png")
