#importing the libraries we need
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn
from utils import sobelOps



class OFFModel (nn.Module):

    def __init__ (self , output_features , seq_size , device) :

        super().__init__()

        #the constants
        self.output_features = output_features
        self.seq_size = seq_size #the seq_size of any input must be equal
        self.batch_size = None #determined as soon as entering the input images
        self.device = device


        # Feature Generation Sub-network
        self.conv1a = nn.Conv2d(3,16,(3,3) , padding = 2)
        self.relu1a = nn.ReLU(inplace = True)
        self.conv1b = nn.Conv2d(16,16,(3,3) , padding = 1)
        self.relu1b = nn.ReLU(inplace = True)
        self.maxpool1 = nn.MaxPool2d((2,2) , padding = 1)

        self.conv2a = nn.Conv2d(16,32,(3,3) )
        self.relu2a = nn.ReLU(inplace = True)
        self.conv2b = nn.Conv2d(32,32,(3,3) , padding = 1)
        self.relu2b = nn.ReLU(inplace = True)
        self.maxpool2 = nn.MaxPool2d((2,2) , padding = 1)

        self.conv3a = nn.Conv2d(32,64,(3,3))
        self.relu3a = nn.ReLU(inplace = True)
        self.conv3b = nn.Conv2d(64,64,(3,3) , padding = 1)
        self.relu3b = nn.ReLU(inplace = True)
        self.conv3c = nn.Conv2d(64,64,(3,3) , padding = 1)
        self.relu3c = nn.ReLU(inplace = True)
        self.maxpool3 = nn.MaxPool2d((2,2) , padding = 1)

        self.conv4a = nn.Conv2d(64,64,(3,3))
        self.relu4a = nn.ReLU(inplace = True)
        self.conv4b = nn.Conv2d(64,64,(3,3) , padding = 1)
        self.relu4b = nn.ReLU(inplace = True)
        self.avgpool4 = nn.AvgPool2d((2,2) , padding = 1)

        self.ffnn5 = nn.Linear(4096, 1024)
        self.relu5 = nn.ReLU(inplace = True)
        self.drop5 = nn.Dropout(p = 0.3)

        self.ffnn6 = nn.Linear(1024 , 101)
        self.softmax = nn.LogSoftmax(dim = 1)

        #OFF Sub-network unit one
        self.off1conv1 = nn.Conv2d(16 , 64 , (1,1)) #for dimentionality control
        self.off1relu1 = nn.ReLU(inplace = True)

        self.off1conv2 = nn.Conv2d(192 , 64 , (1,1) )
        self.off1relu2 = nn.ReLU(inplace = True)

        self.off1conv3 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.off1relu3 = nn.ReLU(inplace = True)
        self.off1conv4 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.off1relu4 = nn.ReLU(inplace = True)
        self.off1conv5 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.off1relu5 = nn.ReLU(inplace = True)

        self.off1avgpool = nn.AvgPool2d((2,2), padding = 1)

        #OFF Sub-network unit two
        self.off2conv1 = nn.Conv2d(32 , 64 , (1,1)) #for dimentionality control
        self.off2relu1 = nn.ReLU(inplace = True)

        self.off2pad = nn.ZeroPad2d(1)

        self.off2conv2 = nn.Conv2d(256 , 64 , (1,1) )
        self.off2relu2 = nn.ReLU(inplace = True)

        self.off2conv3 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.off2relu3 = nn.ReLU(inplace = True)
        self.off2conv4 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.off2relu4 = nn.ReLU(inplace = True)
        self.off2conv5 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.off2relu5 = nn.ReLU(inplace = True)

        self.off2avgpool = nn.AvgPool2d((2,2))

        #OFF Sub-network unit three
        self.off3conv1 = nn.Conv2d(64 , 64 , (1,1)) #for dimentionality control
        self.off3relu1 = nn.ReLU(inplace = True)

        self.off3pad = nn.ZeroPad2d(1)

        self.off3conv2 = nn.Conv2d(256 , 64 , (1,1) )
        self.off3relu2 = nn.ReLU(inplace = True)
        self.off3conv3 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.off3relu3 = nn.ReLU(inplace = True)
        self.off3conv4 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.off3relu4 = nn.ReLU(inplace = True)
        self.off3conv5 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.off3relu5 = nn.ReLU(inplace = True)

        self.off3avgpool = nn.AvgPool2d((2,2))

        self.offFinconv1 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.offFinrelu1 = nn.ReLU(inplace = True)
        self.offFinconv2 = nn.Conv2d(64 , 64 , (3,3) , padding= 1)
        self.offFinrelu2 = nn.ReLU(inplace = True)

        self.offFinavgpool = nn.AvgPool2d((2,2))

        self.off_ffnn5 = nn.Linear(4096, 1024)
        self.off_relu5 = nn.ReLU(inplace = True)
        self.off_drop5 = nn.Dropout(p = 0.3)

        self.off_ffnn6 = nn.Linear(1024 , 101)
        self.off_softmax = nn.LogSoftmax(dim = 1)



    def forward (self , x) :

        # x shape should be like (batch_size*seq_size , 3 , 124 , 124)

        self.batch_size = int(x.shape[0] / self.seq_size) #calculating the batch size

        # Feature Generation Sub-network
        x1 = self.conv1a(x)
        x1 = self.relu1a(x1)
        x1 = self.conv1b(x1)
        x1 = self.relu1b(x1)

        x1_pooled = self.maxpool1(x1)

        x2 = self.conv2a(x1_pooled)
        x2 = self.relu2a(x2)
        x2 = self.conv2b(x2)
        x2 = self.relu2b(x2)

        x2_pooled = self.maxpool2(x2)

        x3 = self.conv3a(x2_pooled)
        x3 = self.relu3a(x3)
        x3 = self.conv3b(x3)
        x3 = self.relu3b(x3)
        x3 = self.conv3c(x3)
        x3 = self.relu3c(x3)

        x3_pooled = self.maxpool3(x3)

        x4 = self.conv4a(x3_pooled)
        x4 = self.relu4a(x4)
        x4 = self.conv4b(x4)
        x4 = self.relu4b(x4)

        x4 = self.avgpool4(x4)

        x4 = x4.view(x4.shape[0] , -1)

        x4 = self.ffnn5(x4)
        x4 = self.relu5(x4)
        x4 = self.drop5(x4)
        x4 = self.ffnn6(x4)
        x4 = self.softmax(x4)
        x4 = x4.view(self.batch_size , self.seq_size , 101)
        x4 = x4.mean(dim = 1)



        #-----------------------------------------------------------------------------------------------------

        #OFF Sub-network unit one

        x1 = self.off1conv1(x1)
        x1 = self.off1relu1(x1)


        #calculating the temporal gradient
        cur_channel_size = x1.shape[1]
        x1_bybatch = x1.view (self.batch_size , -1 , 126 , 126) #the shape is (batch_size , seq_len * channel_size , h,w)

        x1_last  = x1_bybatch[:, cur_channel_size : , : , :] #all the frames except the first one
        x1_first  = x1_bybatch [:, : -cur_channel_size, : , :] #all the frames except the last one

        x1T_reshaped = x1_last - x1_first #substracting the consective frames from each other
        x1T = x1T_reshaped.view(-1 , cur_channel_size , 126 , 126) #the shape is (batch_size* (seq_len-1) , channel_size , h,w)


        #computing the spatial gradient
        x1_spatial = x1[: self.batch_size * (self.seq_size -1), :, : , :] #discarding the last frame
        x1XY = sobelOps(x1_spatial , self.device) #the shape is (batch_size* (seq_len-1) , channel_size , h,w)

        #stacking the gradients
        x1XYT = torch.cat([x1XY , x1T] , dim = 1) #shape (batch_size* (seq_len-1) , 3*channel_size , h,w)

        x1_off1_red = self.off1conv2(x1XYT)
        x1_off1_red = self.off1relu2(x1_off1_red)

        x1_off1_out = self.off1conv3(x1_off1_red)
        x1_off1_out = self.off1relu3(x1_off1_out)
        x1_off1_out = self.off1conv4(x1_off1_out)
        x1_off1_out = self.off1relu4(x1_off1_out)
        x1_off1_out = self.off1conv5(x1_off1_out)
        x1_off1_out = self.off1relu5(x1_off1_out)

        x1_off1_output = x1_off1_out + x1_off1_red


        x1_off1_output_pooled = self.off1avgpool(x1_off1_output)


        #-----------------------------------------------------------------------------------------------------

        #OFF Sub-network unit two

        x2 = self.off2conv1(x2)
        x2 = self.off2relu1(x2)

        #calculating the temporal gradient
        cur_channel_size = x2.shape[1]
        x2_bybatch = x2.view (self.batch_size , -1 , 62 , 62) #the shape is (batch_size , seq_len * channel_size , h,w)

        x2_last  = x2_bybatch[:, cur_channel_size : , : , :] #all the frames except the first one
        x2_first  = x2_bybatch [:, : -cur_channel_size, : , :] #all the frames except the last one

        x2T_reshaped = x2_last - x2_first #substracting the consective frames from each other
        x2T = x2T_reshaped.view(-1 , cur_channel_size , 62 , 62) #the shape is (batch_size* (seq_len-1) , channel_size , h,w)


        #computing the spatial gradient
        x2_spatial = x2[: self.batch_size * (self.seq_size -1), :, : , :] #discarding the last frame
        x2XY = sobelOps(x2_spatial , self.device) #the shape is (batch_size* (seq_len-1) , channel_size , h,w)

        #stacking the gradients
        x2XYT = torch.cat([x2XY , x2T ] , dim = 1) #shape (batch_size* (seq_len-1) , 3*channel_size , h,w)
        x2XYT = self.off2pad(x2XYT)


        x2XYT = torch.cat([x2XYT , x1_off1_output_pooled ] , dim = 1) #shape (batch_size* (seq_len-1) , 3*channel_size + prev_chan_size , h,w)

        x2_off2_red = self.off2conv2(x2XYT)
        x2_off2_red = self.off2relu2(x2_off2_red)

        x2_off2_out = self.off2conv3(x2_off2_red)
        x2_off2_out = self.off2relu3(x2_off2_out)
        x2_off2_out = self.off2conv4(x2_off2_out)
        x2_off2_out = self.off2relu4(x2_off2_out)
        x2_off2_out = self.off2conv5(x2_off2_out)
        x2_off2_out = self.off2relu5(x2_off2_out)


        x2_off2_output = x2_off2_out + x2_off2_red

        x2_off2_output_pooled = self.off2avgpool(x2_off2_output)

        #-----------------------------------------------------------------------------------------------------


        #OFF Sub-network unit three

        x3 = self.off3conv1(x3)
        x3 = self.off3relu1(x3)

        #calculating the temporal gradient
        cur_channel_size = x3.shape[1]
        x3_bybatch = x3.view (self.batch_size , -1 , 30 , 30) #the shape is (batch_size , seq_len * channel_size , h,w)

        x3_last  = x3_bybatch[:, cur_channel_size : , : , :] #all the frames except the first one
        x3_first  = x3_bybatch [:, : -cur_channel_size, : , :] #all the frames except the last one

        x3T_reshaped = x3_last - x3_first #substracting the consective frames from each other
        x3T = x3T_reshaped.view(-1 , cur_channel_size , 30 , 30) #the shape is (batch_size* (seq_len-1) , channel_size , h,w)


        #computing the spatial gradient
        x3_spatial = x3[: self.batch_size * (self.seq_size -1), :, : , :] #discarding the last frame
        x3XY = sobelOps(x3_spatial , self.device) #the shape is (batch_size* (seq_len-1) , channel_size , h,w)

        #stacking the gradients
        x3XYT = torch.cat([x3XY , x3T] , dim = 1) #shape (batch_size* (seq_len-1) , 3*channel_size , h,w)
        x3XYT = self.off3pad(x3XYT)

        x3XYT = torch.cat([x3XYT , x2_off2_output_pooled ] , dim = 1) #shape (batch_size* (seq_len-1) , 3*channel_size + prev_chan_size , h,w)

        x3_off3_red = self.off3conv2(x3XYT)
        x3_off3_red = self.off3relu2(x3_off3_red)

        x3_off3_out = self.off3conv3(x3_off3_red)
        x3_off3_out = self.off3relu3(x3_off3_out)
        x3_off3_out = self.off3conv4(x3_off3_out)
        x3_off3_out = self.off3relu4(x3_off3_out)
        x3_off3_out = self.off3conv5(x3_off3_out)
        x3_off3_out = self.off3relu5(x3_off3_out)

        x3_off3_output = x3_off3_out + x3_off3_red

        x3_off3_output_pooled = self.off3avgpool(x3_off3_output)

        final_off_output = self.offFinconv1(x3_off3_output_pooled)
        final_off_output = self.offFinrelu1(final_off_output)
        final_off_output = self.offFinconv2(final_off_output)
        final_off_output = self.offFinrelu2(final_off_output)

        final_off_output = self.offFinavgpool(final_off_output)

        final_off_output = final_off_output.view(final_off_output.shape[0] , -1)

        final_off_output = self.off_ffnn5(final_off_output)
        final_off_output = self.off_relu5(final_off_output)
        final_off_output = self.off_drop5(final_off_output)
        final_off_output = self.off_ffnn6(final_off_output)
        final_off_output = self.off_softmax(final_off_output)

        final_off_output = final_off_output.view(self.batch_size , self.seq_size - 1  , 101)
        final_off_output = final_off_output.mean(dim = 1)

        output = (final_off_output + x4)/2

        return output
