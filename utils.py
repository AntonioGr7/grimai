from matplotlib import pyplot as plt

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def read_data():
        #TO DO
        xtrain = None
        ytrain = None
        xvalid = None
        yvalid = None
        xtesting = None
        ytesting = None
        return (xtrain,ytrain),(xvalid,yvalid),(xtesting,ytesting)

    @staticmethod
    def plot_losses_graph(training_losses,validation_losses):
        plt.plot(range(training_losses), training_losses, 'g', label='Training loss')
        plt.plot(range(validation_losses), validation_losses, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    @staticmethod
    def save_losses_graph(training_losses,validation_losses,path):
        plt.plot(range(training_losses), training_losses, 'g', label='Training loss')
        plt.plot(range(validation_losses), validation_losses, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path)