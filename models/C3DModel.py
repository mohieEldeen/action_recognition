#importing the libraries we need
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn

class C3DModel (nn.Module):

    def __init__(self , output_features):

        super().__init__()

        self.output_features = output_features
        #the expected input is 3 X 16 X 128 X 128 >>> C x T x H x W


        self.conv1a = nn.Conv3d(3 , 64 , (3,3,3) , padding = 1)     # 64 X 16 X 128  X 128
        self.bn1a = nn.BatchNorm3d(64)
        self.elu1a = nn.ELU(inplace = True)
        self.conv1b = nn.Conv3d(64 , 64 , (3,3,3) , padding = 1)    # 64 X 16 X 128  X 128
        self.elu1b = nn.ELU(inplace = True)
        self.avgpool1 = nn.AvgPool3d((1,2,2))                       # 64 X 16 X 64  X 64


        self.conv2a = nn.Conv3d(64 , 128 , (3,3,3) , padding = 1)   # 128 X 16 X 64  X 64
        self.bn2a = nn.BatchNorm3d(128)
        self.elu2a = nn.ELU(inplace = True)
        self.conv2b = nn.Conv3d(128 , 128 , (3,3,3) , padding = 1)  # 128 X 16 X 64  X 64
        self.elu2b = nn.ELU(inplace = True)
        self.avgpool2 = nn.AvgPool3d(2)                             # 128 X 8 X 32  X 32


        self.conv3a = nn.Conv3d(128 , 256 , (3,3,3) , padding = 1)  # 256 X 8 X 32  X 32
        self.bn3a = nn.BatchNorm3d(256)
        self.elu3a = nn.ELU(inplace = True)
        self.conv3b = nn.Conv3d(256 , 256 , (3,3,3) , padding = 1)  # 256 X 8 X 32  X 32
        self.elu3b = nn.ELU(inplace = True)
        self.conv3c = nn.Conv3d(256 , 256 , (1,1,1) )               # 256 X 8 X 32  X 32
        self.elu3c = nn.ELU(inplace = True)
        self.avgpool3 = nn.AvgPool3d(2)                             # 256 X 4 X 16  X 16


        self.conv4a = nn.Conv3d(256 , 512 , (3,3,3) , padding = 1)  # 512 X 4 X 16  X 16
        self.bn4a = nn.BatchNorm3d(512)
        self.elu4a = nn.ELU(inplace = True)
        self.conv4b = nn.Conv3d(512 , 512 , (3,3,3) , padding = 1)  # 512 X 4 X 16  X 16
        self.elu4b = nn.ELU(inplace = True)
        self.conv4c = nn.Conv3d(512 , 512 , (1,1,1) )               # 512 X 4 X 16 X 16
        self.elu4c = nn.ELU(inplace = True)
        self.avgpool4 = nn.AvgPool3d(2)                             # 512 X 2 X 8  X 8


        self.conv5a = nn.Conv3d(512 , 512 , (3,3,3) , padding = 1)  # 512 X 2 X 8  X 8
        self.bn5a = nn.BatchNorm3d(512)
        self.elu5a = nn.ELU(inplace = True)
        self.conv5b = nn.Conv3d(512 , 512 , (3,3,3) , padding = 1)  # 512 X 2 X 8  X 8
        self.elu5b = nn.ELU(inplace = True)
        self.conv5c = nn.Conv3d(512 , 512 , (1,1,1) )               # 512 X 2 X 8  X 8
        self.elu5c = nn.ELU(inplace = True)
        self.avgpool5 = nn.AvgPool3d(2)                             # 512 X 1 X 4  X 4


        self.h1 = nn.Linear(8192 , 2048)
        self.eluh1 = nn.ELU(inplace = True)
        self.droput1 = nn.Dropout(p = 0.4)

        self.h2 = nn.Linear(2048 , 1024)
        self.eluh2 = nn.ELU(inplace = True)
        self.droput2 = nn.Dropout(p = 0.4)

        self.h3 = nn.Linear(1024 , self.output_features)
        self.logsoftmax = nn.LogSoftmax(dim = 1)



    def forward(self , x):

        #x should be like (n , 3 , 16 , 128 , 128)

        


        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.elu1a(x)
        x = self.conv1b(x)
        x = self.elu1b(x)


        x = self.avgpool1(x)


        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.elu2a(x)
        x = self.conv2b(x)
        x = self.elu2b(x)

        x = self.avgpool2(x)


        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.elu3a(x)
        x = self.conv3b(x)
        x = self.elu3b(x)
        x = self.conv3c(x)
        x = self.elu3c(x)

        x = self.avgpool3(x)


        x = self.conv4a(x)
        x = self.bn4a(x)
        x = self.elu4a(x)
        x = self.conv4b(x)
        x = self.elu4b(x)
        x = self.conv4c(x)
        x = self.elu4c(x)

        x = self.avgpool4(x)


        x = self.conv5a(x)
        x = self.bn5a(x)
        x = self.elu5a(x)
        x = self.conv5b(x)
        x = self.elu5b(x)
        x = self.conv5c(x)
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
