#importing the libraries we need
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn


class FFNNModel (nn.Module):

    def __init__ (self , input_features , output_features):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.h1 = nn.Linear(self.input_features , 2048)
        self.activ1 = nn.ReLU(inplace = True)
        self.drop1 = nn.Dropout(p = 0.3)

        self.h2 = nn.Linear(2048 , 1024)
        self.activ2 = nn.ReLU(inplace = True)
        self.drop2 = nn.Dropout(p = 0.4)

        self.h3 = nn.Linear(1024 , self.output_features)
        self.logsoftmax = nn.LogSoftmax(dim = 1)


    def forward(self ,x) :

        x = self.h1(x)
        x = self.activ1(x)
        x = self.drop1(x)

        x = self.h2(x)
        x = self.activ2(x)
        x = self.drop2(x)

        x = self.h3(x)
        x = self.logsoftmax(x)

        return x
