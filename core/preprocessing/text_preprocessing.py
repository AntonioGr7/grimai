from io import StringIO
from html.parser import HTMLParser
from sklearn.model_selection import train_test_split

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def remove_html(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

class StratifiedSplit():
    @staticmethod
    def data_split(x,y,test_size=0.20,seed=1283,shuffle=False,stratified=True):
        if stratified:
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=seed, shuffle=True,stratify=y)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed,
                                                                shuffle=shuffle, stratify=None)
        return (x_train,y_train),(x_test,y_test)

def split(x,y,test_size=0.20,seed=1283,shuffle=False,stratify=True):
    return StratifiedSplit().data_split(x,y,test_size,seed,shuffle,stratify)