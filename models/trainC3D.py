#importing the libraries we need
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.autograd import Variable


def trainC3D (model , lossFunc, optimizer , epochs , train_loader , val_loader , device):

    val_data_itr = iter(val_loader)

    tr_loss_lis = []
    tr_acc_lis = []
    val_loss_lis = []
    val_acc_lis = []

    for epoch in range (epochs): # looping over the epochs


        tr_running_loss = 0
        tr_tot_acc = 0

        val_running_loss = 0
        val_tot_acc = 0


        for i , data in enumerate(train_loader):

            model.train()

            images = data['image']
            labels = data['label']

            images = Variable(images, requires_grad = False).to(device)
            labels = Variable(labels , requires_grad = False).to(device)

            labels -= 1


            optimizer.zero_grad()

            outputs = model(images) #forward + backward + optimisation

            loss = lossFunc(outputs , labels)

            loss.backward()

            optimizer.step()

            with torch.no_grad():

                tr_running_loss += loss.item()

                _ , predicted = torch.max(outputs, 1)
                c = (predicted == labels)
                tr_acc = (c.sum() / len(c)).item()
                tr_tot_acc  += tr_acc

                model.eval()

                try :
                    val_data = next(val_data_itr)
                except:
                    val_data_itr = iter(val_loader)
                    val_data = next(val_data_itr)


                val_images = val_data['image']
                val_labels = val_data['label']

                val_images = Variable(val_images , requires_grad = False).to(device)
                val_labels = Variable(val_labels , requires_grad = False).to(device)

                val_labels -= 1

                val_outputs = model(val_images) #forward only to get the val outputs
                val_loss = lossFunc(val_outputs , val_labels)
                val_running_loss += val_loss.item()

                _ , val_predicted = torch.max(val_outputs, 1)
                val_c = (val_predicted == val_labels)
                val_acc = (val_c.sum() / len(val_c)).item()
                val_tot_acc  += val_acc




            if (i+1)% 10 == 0:
                print(f'[{epoch+1}, {i+1}] train_loss:{tr_running_loss / 10 : .5f} train_acc:{100 * tr_tot_acc / 10 : 0.5f} val_loss:{val_running_loss / 10 : 0.5f} val_acc:{100 * val_tot_acc / 10 : 0.5f}' )

                tr_loss_lis.append(tr_running_loss)
                tr_acc_lis.append(tr_tot_acc)

                val_loss_lis.append(val_running_loss)
                val_acc_lis.append(val_tot_acc)

                tr_tot_acc = 0
                tr_running_loss = 0.0

                val_tot_acc = 0
                val_running_loss = 0.0


            if i == 0:
                print(f'[{epoch+1}, {i+1}] train_loss:{tr_running_loss  : .5f} train_acc:{100 * tr_tot_acc  : 0.5f} val_loss:{val_running_loss : 0.5f} val_acc:{100 * val_tot_acc  : 0.5f}' )

                tr_loss_lis.append(tr_running_loss)
                tr_acc_lis.append(tr_tot_acc)

                val_loss_lis.append(val_running_loss)
                val_acc_lis.append(val_tot_acc)





    print('Finished Runing')

    return tr_loss_lis, tr_acc_lis, val_loss_lis, val_acc_lis
