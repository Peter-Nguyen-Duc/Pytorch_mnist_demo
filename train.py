


## Imports
import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

#%matplotlib inline
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F


import torch.nn as nn


def get_data_loaders(device):
    """
    The original guide does not support GPU, so i looked somewhere else and used their implemetation instead:
    https://parcc.upenn.edu/training/getting-started/zero-to-mnist/
    """
    train_ds = MNIST(root='data/', train=True,  download=True, transform=transforms.ToTensor())
    test_ds  = MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())

    # stack entire dataset into single tensors, move once
    X_train = torch.stack([img for img, _ in train_ds]).to(device)   
    y_train = torch.tensor([lbl for _, lbl in train_ds]).to(device)
    X_test  = torch.stack([img for img, _ in test_ds]).to(device)
    y_test  = torch.tensor([lbl for _, lbl in test_ds]).to(device)
    return X_train, y_train, X_test, y_test




def fit(epochs, lr, model, X, y, X_val, y_val, batch_size, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    n = X.shape[0]


    for epoch in range(epochs):
        perm = torch.randperm(n, device=X.device)          # shuffle on GPU

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            loss = model.training_step((X[idx], y[idx]))

            # this performs back propagation across the gradients acummulated while running pytorch.
            # It figures out which direction to go, to minimize the loss in the next run
            loss.backward()

            # This method updates the weights of the neural network model correlated to this optimizer.
            # Based on the loss function that was back propagated
            optimizer.step()

            # This method cleans out all the gradients, such that old gradients from previous model versions
            # does not affect the new training step, essentially you reset the accumulated gradients.
            optimizer.zero_grad()


        
        ## Validation phase
        with torch.no_grad():
            result = model.validation_epoch_end(X, y, X_val, y_val)
            model.epoch_end(epoch, result)
            history.append(result)


    return(history)




class MnistModel(nn.Module):
    """
    Class directly yoinked from: "https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch"
    """
    def __init__(self, input_size, num_classes, hidden_layer_size):
        """
        For demosntration purposes i use multiple network designs

        model 1:    Linear model that came with the source: "https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch"
                    Works surprisingly well, but can get higher accuracy with a non-linear model (activation functions in neural networks)

        model 2:    Some shit i cooked up for fun. This model contains around all activation functions i know, a Resnet connection
                    for extra measure, and small descriptions of all the activation functions. WARNING: This is not how you design
                    a good network, usually you stick with 1 type of activation function depending on the output range needs.
        
        model 3:    this model overfits the fuck outta the mnist dataset, which i require to demonstrate the power of 
                    regularization to counter network overfitting the source: "https://stackoverflow.com/questions/71642179/mnist-overfitting"
                    
        """

        super().__init__()

        # MODEL 1
        # Cringe network that came with the tutorial 
        self.linear = nn.Linear(input_size, num_classes)

        # MODEL 2
        # cool peter network
        self.fa1 = nn.Linear(input_size, hidden_layer_size)
        self.fa2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fa3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fa4 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.resnet_dimension_sync_layer = nn.Linear(hidden_layer_size, input_size)
        self.fa5 = nn.Linear(input_size, num_classes)

        # MODEL 3
        # Overfitting network
        self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,4096),
        nn.ReLU(),
        nn.Linear(4096,2048),
        nn.ReLU(),
        nn.Linear(2048,1024),
        nn.ReLU(),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Linear(512,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Linear(16,10))


    def forward(self, x):
        """
        SUMMARY: 
        When the model is called, this is the function that is run. Since this class inherits the properties of
        nn-module, then it inherits an overloading property, which calls forward when you call this class using:

        model = MnistModel(some inputs)
        model() <--- model call
        """


        #  ---- MODEL 1: Cringe linear version with maximum accuracy of around 85%
        # came from the source that i yoinked the code from 
        # xb = xb.reshape(-1, 784)
        # out = self.linear(xb)


        # ----- MODEL 2: Network that uses all activation functions and weird tricks i know
        # NOTE: I dont have any practical reason to use a different activation function for each class,
        # Its just for style points :D
        # x = x.reshape(-1, 784)

        # # Creating a resnet connection for fun
        # x_resnet = x

        # # Tanh limits all numbers between -1, 1
        # x = torch.tanh(self.fa1(x))


        # # Sigmoid limits all numbers between 0, 1
        # x = torch.sigmoid(self.fa2(x))


        # # relu limits all numbers between 0 and infinite         
        # x = torch.relu(self.fa3(x))


        # # Leaky relu limits all numbers between technically -infinite to infinite. 
        # # But it is very difficult to get a high negative value. 
        # x = torch.nn.functional.leaky_relu(self.fa4(x))

        # # converts hidden layer input data dimensions back to input layer data dimension size
        # x = self.resnet_dimension_sync_layer(x)

        # # Residual connection
        # x = x + x_resnet


        # # softmax limits the values between 0 and 1, 
        # # But also enforces that the sum of all outputs are 1
        # x = torch.softmax(self.fa5(x), dim=1)


        # ---- MODEL 3:  network i found in the internet that overfits the mnist dataset 
        # source: https://stackoverflow.com/questions/71642179/mnist-overfitting

        x = self.layers(x)

        return(x)




    
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim = 1)
        return (preds == labels).sum() / len(preds) 

        
    def training_step(self, batch):
        images, labels = batch
        images = images
        labels = labels
        out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels) ## Calculate the loss
        return(loss)

    
    def validation_epoch_end(self, outputs_train, labels_train, outputs_val, labels_val):

        outputs_train = self(outputs_train)
        loss_train = F.cross_entropy(outputs_train, labels_train)
        acc_train = self.accuracy(outputs_train, labels_train)


        outputs_val = self(outputs_val)
        loss_val = F.cross_entropy(outputs_val, labels_val)
        acc_val = self.accuracy(outputs_val, labels_val)


        
        # Use .item() to convert tensors to python floats
        return {
            "train_loss": loss_train.item(), 
            "train_acc": acc_train.item(),
            "val_loss": loss_val.item(), 
            "val_acc": acc_val.item()
        }

    
    
    def epoch_end(self, epoch,result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, train_acc: {result['train_acc']:.4f} -- val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

def main():

    # This variable represents the GPU if available, and is passed to all training steps to get GPU accelerated training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train, X_test, y_test = get_data_loaders(device)

    input_size = 28 * 28
    hidden_layer_size = 512
    num_classes = 10


    model = MnistModel(input_size=input_size, num_classes=num_classes, hidden_layer_size=hidden_layer_size).to(device)

    print("="*20)
    print("Model initial performance performance: ")
    result = model.validation_epoch_end(X_train, y_train, X_test, y_test)
    model.epoch_end(0, result)
    print("="*20)
    print("\n"*2)

    # To hit 

    hist_list = fit(epochs=300, lr=5e-4, model=model,X=X_train, y=y_train, X_val=X_test, y_val=y_test, batch_size=8192, opt_func=torch.optim.Adam)




   # 1. Extract all values from history
    train_accs = [result['train_acc'] for result in hist_list]
    val_accs = [result['val_acc'] for result in hist_list]
    train_losses = [result['train_loss'] for result in hist_list]
    val_losses = [result['val_loss'] for result in hist_list]

    # 2. Create a figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # --- TOP PLOT: ACCURACY ---
    ax1.plot(train_accs, color="blue", label="Train Accuracy")
    ax1.plot(val_accs, color="red", label="Val Accuracy")
    ax1.set_title('Accuracy Vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- BOTTOM PLOT: LOSS ---
    ax2.plot(train_losses, color="blue", label="Train Loss")
    ax2.plot(val_losses, color="red", label="Val Loss")
    ax2.set_title('Loss Vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Adjust layout so titles and labels don't overlap
    plt.tight_layout()
    plt.show()



if __name__=="__main__":

    main()