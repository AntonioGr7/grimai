from matplotlib import pyplot as plt

class Plotter():
    def __init__(self):
        pass
    def plot_losses(self,train_losses,valid_losses):
        plt.plot(range(len(train_losses)), train_losses, 'g', label='Training loss')
        plt.plot(range(len(valid_losses)), valid_losses, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        #plt.savefig(path)