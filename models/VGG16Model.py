#importing the libraries we need
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn

class CNNModel (nn.Module):

    def __init__(self , output_features):

        super().__init__()

        self.output_features = output_features
        #the expected input is 3 X 224 X 224

        self.conv1a = nn.Conv2d(3 , 64 , (3,3) , padding = 1)     # 64 X 224  X 224
        self.bn1a = nn.BatchNorm2d(64)
        self.elu1a = nn.ELU(inplace = True)
        self.conv1b = nn.Conv2d(64 , 64 , (3,3) , padding = 1)    # 64 X 224  X 224
        self.bn1b = nn.BatchNorm2d(64)
        self.elu1b = nn.ELU(inplace = True)
        self.maxpool1 = nn.AvgPool2d(2)                           # 64 X 112  X 112


        self.conv2a = nn.Conv2d(64 , 128 , (3,3) , padding = 1)   # 128 X 112  X 112
        self.bn2a = nn.BatchNorm2d(128)
        self.elu2a = nn.ELU(inplace = True)
        self.conv2b = nn.Conv2d(128 , 128 , (3,3) , padding = 1)  # 128 X 112  X 112
        self.bn2b = nn.BatchNorm2d(128)
        self.elu2b = nn.ELU(inplace = True)
        self.maxpool2 = nn.AvgPool2d(2)                           # 128 X 56  X 56


        self.conv3a = nn.Conv2d(128 , 256 , (3,3) , padding = 1)  # 256 X 56  X 56
        self.bn3a = nn.BatchNorm2d(256)
        self.elu3a = nn.ELU(inplace = True)
        self.conv3b = nn.Conv2d(256 , 256 , (3,3) , padding = 1)  # 256 X 56  X 56
        self.bn3b = nn.BatchNorm2d(256)
        self.elu3b = nn.ELU(inplace = True)
        self.conv3c = nn.Conv2d(256 , 256 , (1,1) )               # 256 X 56  X 56
        self.bn3c = nn.BatchNorm2d(256)
        self.elu3c = nn.ELU(inplace = True)
        self.maxpool3 = nn.AvgPool2d(2)                           # 256 X 28  X 28


        self.conv4a = nn.Conv2d(256 , 512 , (3,3) , padding = 1)  # 512 X 28  X 28
        self.bn4a = nn.BatchNorm2d(512)
        self.elu4a = nn.ELU(inplace = True)
        self.conv4b = nn.Conv2d(512 , 512 , (3,3) , padding = 1)  # 512 X 28  X 28
        self.bn4b = nn.BatchNorm2d(512)
        self.elu4b = nn.ELU(inplace = True)
        self.conv4c = nn.Conv2d(512 , 512 , (1,1) )               # 512 X 28  X 28
        self.bn4c = nn.BatchNorm2d(512)
        self.elu4c = nn.ELU(inplace = True)
        self.maxpool4 = nn.AvgPool2d(2)                           # 512 X 14  X 14

        self.conv5a = nn.Conv2d(512 , 512 , (3,3) , padding = 1)  # 512 X 14  X 14
        self.bn5a = nn.BatchNorm2d(512)
        self.elu5a = nn.ELU(inplace = True)
        self.conv5b = nn.Conv2d(512 , 512 , (3,3) , padding = 1)  # 512 X 14  X 14
        self.bn5b = nn.BatchNorm2d(512)
        self.elu5b = nn.ELU(inplace = True)
        self.conv5c = nn.Conv2d(512 , 512 , (1,1) )               # 512 X 14  X 14
        self.bn5c = nn.BatchNorm2d(512)
        self.elu5c = nn.ELU(inplace = True)
        self.avgpool5 = nn.AvgPool2d(2)                           # 512 X 7  X 7


        self.h1 = nn.Linear(25088 , 2048)
        self.eluh1 = nn.ELU(inplace = True)
        self.droput1 = nn.Dropout(p = 0.4)

        self.h2 = nn.Linear(2048 , 1024)
        self.eluh2 = nn.ELU(inplace = True)
        self.droput2 = nn.Dropout(p = 0.4)

        self.h3 = nn.Linear(1024 , self.output_features)
        self.logsoftmax = nn.LogSoftmax(dim = 1)



    def forward(self , x):

        #x should be like (n , 3 , 224 , 224)


        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.elu1a(x)
        x = self.conv1b(x)
        #x = self.bn1b(x)
        x = self.elu1b(x)
        x = self.maxpool1(x)

        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.elu2a(x)
        x = self.conv2b(x)
        #x = self.bn2b(x)
        x = self.elu2b(x)
        x = self.maxpool2(x)

        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.elu3a(x)
        x = self.conv3b(x)
        #x = self.bn3b(x)
        x = self.elu3b(x)
        x = self.conv3c(x)
        #x = self.bn3c(x)
        x = self.elu3c(x)
        x = self.maxpool3(x)

        x = self.conv4a(x)
        x = self.bn4a(x)
        x = self.elu4a(x)
        x = self.conv4b(x)
        #x = self.bn4b(x)
        x = self.elu4b(x)
        x = self.conv4c(x)
        #x = self.bn4c(x)
        x = self.elu4c(x)
        x = self.maxpool4(x)

        x = self.conv5a(x)
        x = self.bn5a(x)
        x = self.elu5a(x)
        x = self.conv5b(x)
        #x = self.bn5b(x)
        x = self.elu5b(x)
        x = self.conv5c(x)
        #x = self.bn5c(x)
        x = self.elu5c(x)
        x = self.avgpool5(x)

        x = x.flatten(1)

        x = self.h1(x)
        x = self.eluh1(x)
        x = self.droput1(x)

        x = self.h2(x)
        x = self.eluh2(x)
        x = self.droput2(x)

        x = self.h3(x)
        x = self.logsoftmax(x)

        return x
