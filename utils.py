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


def saveFrames (wantedFPS  , VidDir , VidName ,  saveDir ):
    """
    Saves the frames of a given video at the given location and returns the number of frames saved.

    Parameters
    ----------
    wantedFPS : int
        The number of frames per second to save the frames with.

    VidDir : string
        the direction of the video that needs to be processed.

    VidName : string
        the name of the video that needs to be processed.

    saveDir : string
        the desired direction to save the frames (make sure it exists)

    Returns
    -------
    the number of frames saved in the desk.

    """

    VidDirPlusName = VidDir + VidName
    Vid = cv2.VideoCapture(VidDirPlusName)
    fps = Vid.get(cv2.CAP_PROP_FPS) #the real fps of the chosen video

    #making sure that the wantedFPS smaller than the real fps
    assert fps >= wantedFPS , "the wantedFPS should be smaller than the real fps"

    #every time the counter can be divided by the checkmark, a frame will be saved to get the wantedFPS
    checkmark = math.floor(fps / wantedFPS)
    frame_counter = 0
    counter = 0

    while Vid.isOpened():

        ret , frame = Vid.read()

        if ret == False : #if there're no more frames in the video, the loop will break
            break

        counter += 1
        if counter % checkmark == 0 :
            frame_counter += 1
            framename = saveDir + VidName + '_' + str(frame_counter) + '.jpg'

            cv2.imwrite(framename , frame)


    Vid.release()
    return frame_counter


def trainValSpliteFromVideoNames(vali_ratio , train_videos):
    """
    this function is used to split the videos into validation and train data by spliting the video names in random
    (the output is shuffled)

    Args :
        vali_ratio (float) : the ratio by which the data will be split
        train_videos (pandas dataFrame) : the data frame that holds the names and the classes of the videos

    Returns :

        tr_frames : dataframe that holds the new train frames
        val_frames : dataframe that holds the new validation frames

    """

    videos_idx = [i for i in range(len(train_videos))] #creating a list to hold the indeces of the train data
    np.random.shuffle (videos_idx) #shuffling the indeces

    train_num = int( len(train_videos) * (1-vali_ratio)  ) #the size of train data

    tr_videos = train_videos.iloc[  videos_idx[ : train_num]   , :] #assigning the train data by the threshold defined
    val_videos = train_videos.iloc[  videos_idx[train_num : ]   , :] #assigning the validation data by the threshold defined

    tr_videos = tr_videos.reset_index().drop(['index'] , axis= 1) #reseting the indeces to avoid some problems
    val_videos = val_videos.reset_index().drop(['index'] , axis= 1) #reseting the indeces to avoid some problems

    #in this for loop we will make a dataFrame that holds the names of the frames and thier responding classes for the train data
    tr_frames = []

    for i in range (len(tr_videos)):

        vid_name = tr_videos['name'][i]
        vid_class = tr_videos['class'][i]
        vid_frames = tr_videos['num_frames'][i]

        tr_frames += [(vid_name + '_' + str(n) + '.jpg' , vid_class) for n in range(1,vid_frames+1)]

    tr_frames = pd.DataFrame(tr_frames , columns= ['name' , 'class'])


    #in this for loop we will make a dataFrame that holds the names of the frames and thier responding classes for the validation data
    val_frames = []

    for i in range (len(val_videos)):

        vid_name = val_videos['name'][i]
        vid_class = val_videos['class'][i]
        vid_frames = val_videos['num_frames'][i]

        val_frames += [(vid_name + '_' + str(n) + '.jpg' , vid_class) for n in range(1,vid_frames+1)]

    val_frames = pd.DataFrame(val_frames , columns= ['name' , 'class'])

    return tr_frames  ,val_frames



def vali_train_splite(vali_ratio , train_videos):
    """
    this function is used to split the videos into validation and train data by spliting the video names in random
    (the output is shuffled)

    Args :
        vali_ratio (float) : the ratio by which the data will be split
        train_videos (pandas dataFrame) : the data frame that holds the names and the classes of the videos

    Returns :

        tr_frames : dataframe that holds the new train frames
        val_frames : dataframe that holds the new validation frames

    """

    videos_idx = [i for i in range(len(train_videos))]
    np.random.shuffle (videos_idx)

    train_num = int( len(train_videos) * (1-vali_ratio)  )

    tr_videos = train_videos.iloc[  videos_idx[ : train_num]   , :]
    val_videos = train_videos.iloc[  videos_idx[train_num : ]   , :]

    tr_videos = tr_videos.reset_index().drop(['index'] , axis= 1)
    val_videos = val_videos.reset_index().drop(['index'] , axis= 1)

    return tr_videos  ,val_videos


class UCF101DatasetFrames(Dataset):
    """
    A class used to generate UCF101 frames given the frames names, thier label and the direction of the image file.

    """

    def __init__(self, frame_name , img_dir , transform = None , shuffle = False ):
        """
        Args:
            frame_name (pandas DataFrame): the Dataframe that contains the name and the class of the frames.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on the image.

            shuffle (boolean , optional) : optional shuffle on the images before before the output

        """
        self.frame_name = frame_name #reading the info of the frames (names and labels)
        self.img_dir = img_dir #directory of the image folder
        self.transform = transform #the transforms that will be done on every image
        self.frame_idx = np.arange(0 , len(self.frame_name) ) #creating an array that carry the index of every frame by order
        self.shuffle = shuffle

        if self.shuffle == True :
            np.random.shuffle(self.frame_idx) #shuffling the order of the frames if shuffle is True

    def __len__(self):
        return len(self.frame_name) #length of the frames

    def __getitem__(self, idx):


        if torch.is_tensor(idx):
            idx = idx.tolist() #tranforming the type of the scaler to avoid errors :')

        idx = self.frame_idx[idx] #getting the new index of the frames (if shuffle == False there will be no change)

        img_name = self.frame_name['name'][idx] #getting the frame name

        image = cv2.imread(self.img_dir + img_name) # extracting the real frame from its directory

        #transforming the image from BGR to RGB because the god damn open cv is unable to do it by itself :)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #extracting the label of the image
        label = self.frame_name['class'][idx]


        #applying the transforms on the image
        if self.transform:
            image = self.transform(image)

        #combining the image and its labels in a dectionary
        sample = {'image': image, 'label': label}

        return sample


class UCF101DatasetSeqs(Dataset):
    """
    A class used to generate UCF101 sequences given the videos names, thier labels, the sequence of frames and the directory
    of the image file.

    """

    def __init__(self, frame_seq_names , img_dir , transform = None , shuffle = False ):
        """
        Args:
            frame_seq_names (pandas DataFrame): the Dataframe that contains the name and the class of the videoz and the frame
                sequences that will be cut from the video.

            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on the image.

            shuffle (boolean , optional) : optional shuffle on the images before before the output

        """
        self.frame_seq_names = frame_seq_names #reading the info of the sequences (names, labels, length)
        self.img_dir = img_dir #directory of the image folder
        self.transform = transform #the transforms that will be done on every sequence of frames
        self.seq_idx = np.arange(0 , len(self.frame_seq_names) ) #creating an array that carry the index of every sequence by order
        self.shuffle = shuffle

        if self.shuffle == True :
            np.random.shuffle(self.seq_idx) #shuffling the order of the sequences if shuffle is True

    def __len__(self):
        return len(self.frame_seq_names) #length of the sequences

    def __getitem__(self, idx):


        if torch.is_tensor(idx):
            idx = idx.tolist()  #tranforming the type of the scaler to avoid errors :')

        idx = self.seq_idx[idx] #getting the new index of the chosen sequence (if shuffle == False there will be no change)

        vid_name = self.frame_seq_names['name'][idx] #getting the video name

        sequence = self.frame_seq_names['seq'][idx]

        images = np.zeros((len(sequence) , 240 , 320 , 3))

        for i , seq_num in enumerate(sequence):

            if seq_num == 0:
                break

            image = cv2.imread(self.img_dir + vid_name + '_' + str(seq_num) + '.jpg') # extracting the frame from its directory

            #transforming the image from BGR to RGB because the god damn open cv is unable to do it by itself :)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            images[i] = image

        #extracting the label of the images sequence
        label = self.frame_seq_names['class'][idx]


        #applying the transforms on the images
        if self.transform:
            images = self.transform(images)

        #combining the images and its labels in a dectionary
        sample = {'image': images, 'label': label}

        return sample





def getSequences (noFrames , seq_len , overlap = None):

    """
    this Function is used to generate sequences of frames for every video. it's very much like a sliding window

    -----------
    Args:
    -----------
    noFrames(int) : the total number of frames in each video.
    seq_len(int) : the sequence of frames that the videos needed to be splitted into
    overlap (int, optional) : the overlap between every two consecutive sequences

    -----------
    Returns:
    -----------

    seqs (list) : a list that contains the sequences list of the frames of the video


    """

    frame_seq = list(range(1,noFrames+1)) #intializing the frame sequence that will use the slidding window upon

    if not overlap :
        overlap = seq_len -1 #if the overlap is not enteres then the overlap be the biggest possible overlap

    else :
        assert overlap < seq_len , "the overlap parameter can't be bigger than or equal to the sequence length"


    if seq_len > noFrames :

        #if the seq_len is bigger than the number of frames then we will simply pad the frames sequence by zeroes
        #so that the number of frames sequence is equal to the sequence length and get one sequence in the end

        padLength = seq_len - noFrames #the length of padding

        frame_seq += [0 for i in range(padLength)] #adding the padding

        noFrames = len(frame_seq) #recomputing the frames length


    stride = seq_len - overlap #the stride that the sliding window would follow

    numOfSeq = ((noFrames - seq_len) / stride) + 1 #calculating the number of sequences possible

    if numOfSeq != int(numOfSeq): #checking if the numOfseq is an integer number or decimal

        #if it's a single integer then the stride is compatable with the number of frames in the video and we don't need to do anything

        #if the numOfseq is a decimal number then that means that the stride is not compatable with
        #the number of frames in the video. that means we need to pad the end of the frames sequence with zeros
        #so that the length of the frames is compatable with the stride

        padLength = stride - ((noFrames - seq_len) % stride) #the length of padding

        frame_seq += [0 for i in range(padLength)] #adding the padding

        noFrames = len(frame_seq) #recomputing the frames length

        numOfSeq = int( ((noFrames - seq_len) / stride) + 1 ) #recomputing the number of sequences



    #computing the sequences
    seqs = [ ]
    for i in range (0 , noFrames - seq_len + 1 , stride) :

        seqs.append(frame_seq[i : i+seq_len])




    return seqs


def sobelOps (tensor , device ):

    """
    applies sobel X and sobel y on every image and  every channel of tensor
    seperatly. the output is concatenation of both; the output of sobel x
    on tensor and sobel y on tensor

    -------------------------------------------------------------------
    Args:
    -------------------------------------------------------------------
    tensor : a pytorch tensor of shape (images_size , channel_size , H , W )
    device : the device on the process will be done (cpu or cuda)

    --------------------------------------------------------------------
    Returns:
    --------------------------------------------------------------------
    the concatenation of both; the output of sobel x on tensor and
    sobel y on tensor  (the concatenation is done on the channel dimension)

    the output shape is like (images_size , 2* channel_size , H ,W )

    """

    #tensor's shape is like (batch_size , channel_size , h , w) so we just get these info out
    batch_size = tensor.shape[0]
    channel_size = tensor.shape[1]
    feature_h = tensor.shape[2]
    feature_w = tensor.shape[3]

    #reshaping the tensor so we can apply the filter on every channel of every batch seperatly
    #by stacking the channels in the batch dimension
    tensor_reshaped = tensor.view((-1 , 1 , feature_h ,feature_w)) #(batch_size*channel_size , 1 , feature_h , feature_w)


    sobelx = torch.Tensor([[-1,0,1],[-1,0,1],[-1,0,1]])
    sobelx = Variable(sobelx.to(device) , requires_grad = False) #the shape is (3,3)
    sobelx = sobelx.unsqueeze(0).unsqueeze(0) #the shape is (1,1,3,3)

    sobely = torch.Tensor([[1,1,1],[0,0,0],[-1,-1,-1]])
    sobely = Variable(sobely.to(device) , requires_grad = False)  #the shape is (3,3)
    sobely = sobely.unsqueeze(0).unsqueeze(0) #the shape is (1,1,3,3)

    tensorX_reshaped = nn.functional.conv2d(tensor_reshaped , sobelx ) #(batch_size*channel_size  , 1 , feature_h-2 , feature_w-2)

    tensorY_reshaped = nn.functional.conv2d(tensor_reshaped , sobely ) #(batch_size*channel_size  , 1 , feature_h-2 , feature_w-2)

    tensorX = tensorX_reshaped.view((batch_size , channel_size , feature_h-2 ,feature_w-2)) # (batch_size , channel_size , feature_h-2 , feature_w-2)

    tensorY = tensorY_reshaped.view((batch_size , channel_size , feature_h-2 ,feature_w-2)) # (batch_size , channel_size , feature_h-2 , feature_w-2)

    tensorX = nn.functional.pad(tensorX , (1,1,1,1)) #pading tensorX so it returns to the original shape

    tensorY = nn.functional.pad(tensorY , (1,1,1,1)) #pading tensorX so it returns to the original shape

    return torch.cat([tensorX , tensorY] , dim = 1)




def cloneParams (target_model , source_model):

    """
    clones the parameters of a pre-trained model to another new model.
    the function is mainly made to clone the parameters of 2D CNN model into another 3D CNN model that has the same architecture

    =====================
    Args :
    =====================

    target_model : the model that's targeted to update its weights (the 3D model)

    source_model : the pretrained model that we are going to clone its parameter
    """

    for layer in target_model.state_dict().keys():

        if layer in source_model.state_dict().keys():

            if target_model.state_dict()[layer].shape == source_model.state_dict()[layer].shape :

                target_model.state_dict()[layer].data.copy_( source_model.state_dict()[layer] )

            else :

                temp = target_model.state_dict()[layer].shape[2] #the temporal dimension

                for i in range (temp) :

                    target_model.state_dict()[layer][:, :, i, :, :].data.copy_(source_model.state_dict()[layer] / temp)

    return target_model
