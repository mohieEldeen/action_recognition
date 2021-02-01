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



def trainOFF (model , lossFunc, optimizer , epochs , train_loader , validation_loader , device):

    model = model.to(device)
    #lossFunc = lossFunc.to(device)

    tr_loss_lis = []
    tr_acc_lis = []

    val_loss_lis = []
    val_acc_lis = []

    val_load_iter = iter(validation_loader)


    for epoch in range (epochs): # looping over the epochs

        tr_running_loss = 0
        tr_tot_acc = 0


        for i , data in enumerate(train_loader):

            model.train()

            images = data['image'] #shape (batch , 3 , seq ,124 ,124)
            labels = data['label']

            labels -= 1

            images = Variable(images.transpose(1,2).reshape(-1 , 3 , 124 , 124)) #shape (batch * seq , 3 ,124 ,124)
            labels = Variable(labels , requires_grad=False)

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images) #forward + backward + optimization

            loss = lossFunc(outputs , labels)

            loss.backward()

            optimizer.step()

            tr_running_loss += loss.item()



            with torch.no_grad():

                _ , predicted = torch.max(outputs, 1)
                c = (predicted == labels)
                tr_acc = (c.sum() / len(c)).item()
                tr_tot_acc  += tr_acc





            if (i+1)% 10 == 0  :

                model.eval()

                with torch.no_grad():

                    try :

                        val_data = next (val_load_iter)


                    except :

                        val_load_iter = iter(validation_loader)
                        val_data = next (val_load_iter)



                    images_val = val_data['image']
                    labels_val = val_data['label']

                    labels_val -= 1

                    images_val = Variable(images_val.view (-1 , 3 , 124 , 124) , requires_grad=False) #shape (batch * seq , 3 ,124 ,124)
                    labels_val = Variable(labels_val , requires_grad=False)

                    images_val = images_val.to(device)
                    labels_val = labels_val.to(device)


                    outputs_val = model(images_val.to(device)) #forward only

                    loss_val = lossFunc(outputs_val , labels_val).item()


                    _ , predicted_val = torch.max(outputs_val, 1)
                    c = (predicted_val == labels_val)
                    val_acc = (c.sum() / len(c)).item()




                print(f'[{epoch+1}, {i+1}] train_loss:{tr_running_loss / 10 : .5f} train_acc:{100 * tr_tot_acc / 10 : 0.5f} val_loss:{loss_val : .5f} val_acc:{100 * val_acc : 0.5f}' )

                tr_loss_lis.append(tr_running_loss)
                tr_acc_lis.append(tr_tot_acc)

                val_loss_lis.append(loss_val)
                val_acc_lis.append(val_acc)

                tr_tot_acc = 0
                tr_running_loss = 0.0



            if i == 0 :

                print(f'[{epoch+1}, {i+1}] train_loss:{tr_running_loss : .5f} train_acc:{100 * tr_tot_acc : 0.5f}' )

                tr_loss_lis.append(tr_running_loss)
                tr_acc_lis.append(tr_tot_acc)


    print('Finished Runing')

    return tr_loss_lis, tr_acc_lis, val_loss_lis, val_acc_lis
