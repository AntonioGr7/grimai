#### Example on IMDB
#### In order to make this code works you need to download imdb_dataset from this url
#### https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
import os
import pandas as pd
from core.preprocessing.text_preprocessing import remove_html,split
from core.engine.engine import Engine,LRFinder
from examples.imdb_sentiment_analysis.data import Data
from examples.imdb_sentiment_analysis.bert_model import Bert
from examples.imdb_sentiment_analysis.inherited_cbs import InheritedCBS
import torch
import torch.optim as optim



if __name__ == "__main__":
    DATA_BASE_DIR = "./data/imdb/"
    dataset = pd.read_csv(os.path.join(DATA_BASE_DIR,"imdb_dataset.csv"),nrows=None)
    dataset['review'] = dataset['review'].apply(remove_html)
    train,valid = split(x=dataset['review'].values,y=dataset['sentiment'].values,test_size=0.5)

    train_set = Data(train[0],train[1])
    valid_set = Data(valid[0],valid[1])

    BATCH_SIZE = 20
    EPOCHS = 3
    LEARNING_RATE = 5e-6

    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=BATCH_SIZE,
                     shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
                    dataset=valid_set,
                    batch_size=BATCH_SIZE,
                    shuffle=False)

    device = torch.device("cuda:0")
    my_model = Bert()

    optimizer = optim.AdamW(my_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    cbs = InheritedCBS()

    device = [1]
    engine = Engine(model=my_model,optimizer=optimizer,cbs=cbs,fp16=True,parallelize=False,scheduler=None,device=device)
    engine.fit(epochs=EPOCHS,train_dataloader=train_loader,valid_dataloader=valid_loader)