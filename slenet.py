import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score







#Loading from numpy files
X = np.load('drive/My Drive/Copy of s_x_train.npy')
y = np.load('drive/My Drive/Copy of s_y_train.npy')

#Converting to pytorch tensors
X = torch.from_numpy(X)
X = X.permute(0,3,1,2)


# Normalizing image to  0 mean and 1 stddev
def getNormalized(X,s):
    flattened_channels = X.reshape(3,-1)
    channel_mean = flattened_channels.mean(dim = 1)
    channel_stddev = flattened_channels.std(dim = 1)
    preprocess2 = transforms.Compose([
                      transforms.Normalize(channel_mean, channel_stddev)
    ])


    temptwo = torch.tensor([])
    for i in range(X.shape[0]):
        a = preprocess2(X[i])
        temptwo = torch.cat([temptwo, a.reshape(1,3,s,s)])
  
    return temptwo


# Resizing image
def imageSetResize(newSize,X):
    preprocess1 = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(newSize),
                        transforms.ToTensor()])
  
    temp = torch.tensor([])
    for i in range(X.shape[0]):
        a = preprocess1(X[i])
        temp = torch.cat([temp, a.reshape(1,3,newSize,newSize)])

    return temp 


# To shuffle and create batches at the beginning of each epoch
def shuffle_and_batch(X,y,num,bs):
    shuffled_indices = torch.randperm(X.shape[0])
    newX = X[shuffled_indices]
    newY = y[shuffled_indices]

    X_batches = []
    y_batches = []
    for i in range(num):
        X_batches.append(X[i*bs:(i+1)*bs])
        y_batches.append(y_train[i*bs:(i+1)*bs])

    return X_batches, y_batches



#Resizing the images to 32x32 and normalizing
imsize = 32
X_train = imageSetResize(imsize, X_train.float())
X = getNormalized(X.float(),imsize)





# to convert from one hot encodings to labels
def labelize(p):
    labelized_preds = []
    for i in p:
        l = 0. if i[0]>i[1] else 1.
        labelized_preds.append(l)

    return torch.tensor(labelized_preds)






# Model
class lenetModel(nn.Module):
    
    def __init__(self):
        super(lenetModel, self).__init__()
        self.c1 = nn.Conv2d(3,6, kernel_size = 5)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.first_cons6_filterlist = nn.ModuleList([])
        self.second_cons6_filterlist = nn.ModuleList([])
        self.third_cons3_filterlist = nn.ModuleList([])
        self.fourth_last1_filterlist = nn.ModuleList([])
        
        for i in range(6):
            self.first_cons6_filterlist.append(nn.Conv2d(3,1,kernel_size = 5))
        for i in range(6):
            self.second_cons6_filterlist.append(nn.Conv2d(4,1,kernel_size = 5))
        for i in range(3):
            self.third_cons3_filterlist.append(nn.Conv2d(4,1,kernel_size = 5))
        self.fourth_last1_filterlist.append(nn.Conv2d(6,1,kernel_size = 5))
        
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 120, kernel_size = 5)
        self.bn3= nn.BatchNorm2d(120)
        self.ll1 = nn.Linear(120, 84)
        self.ll2 = nn.Linear(84,2)
        
        
    
    def forward(self, x):
        
        nos = x.shape[0]
        ef = lambda x,y: torch.index_select(y,1,torch.tensor(x))
        c1_out = self.c1(x)
        out = F.relu(self.bn1(self.pool1(c1_out)))
        lione = [ef([0,1,2],out),ef([1,2,3],out), ef([2,3,4],out), ef([3,4,5],out), ef([0,4,5],out),ef([0,1,5],out)]
        litwo = [ef([0,1,2,3],out),ef([1,2,3,4],out), ef([2,3,4,5],out), ef([0,3,4,5],out), ef([0,1,4,5],out),ef([0,1,2,5],out)]
        lithree = [ef([0,1,3,4],out),ef([1,2,4,5],out), ef([0,2,3,5],out)]
        lifour = [ef([0,1,2,3,4,5],out)]
        feature_maps1 = []
        feature_maps2 = []
        feature_maps3 = []
        feature_maps4 = []
        for i in range(6):
            feature_maps1.append(self.first_cons6_filterlist[i](lione[i]))
        for i in range(6):
            feature_maps2.append(self.second_cons6_filterlist[i](litwo[i]))
        for i in range(3):
            feature_maps3.append(self.third_cons3_filterlist[i](lithree[i]))
        for i in range(1):
            feature_maps4.append(self.fourth_last1_filterlist[i](lifour[i]))
        fms = []
        fms.extend(feature_maps1)
        fms.extend(feature_maps2)
        fms.extend(feature_maps3)
        tfms = torch.Tensor([])
        for i in fms:
            tfms = torch.cat([tfms, i], dim=1)
        c2_out = F.relu(self.bn2(self.pool2(tfms)))
        c3_out = self.bn3(self.conv3(c2_out))
        c3_out = c3_out.reshape(nos,120)
        ll1_out = F.relu(self.ll1(c3_out))
        ll2_out = self.ll2(ll1_out)
        preds = nn.Softmax(dim=1)(ll2_out)
        return preds






#Loading trained model
model_path = './models/10.pth'
a1 = lenetModel()
n1 = torch.load(model_path)
a1.load_state_dict(n1)
a1.eval()



raw_preds = a1(Xraw)
raw_preds = labelize(raw_preds)
raw_prediction_comparisons = (yraw == raw_preds)
raw_accuracy = float(raw_prediction_comparisons.sum())/float(yraw.shape[0])
print('RawAccuracy:{}'.format(raw_accuracy)) 
print('Raw Dataset Confusion Matrix')
print(confusion_matrix(yraw, raw_preds))
print('Raw Dataset F1-Score: ', end='')
print(f1_score(yraw, raw_preds, average = 'weighted'))
print('0 Accuracy: '+str(confusion_matrix(yraw, raw_preds)[0][0]/confusion_matrix(yraw, raw_preds)[0].sum()*100)+' %')
print('1 Accuracy: '+str(confusion_matrix(yraw, raw_preds)[1][1]/confusion_matrix(yraw, raw_preds)[1].sum()*100)+' %')
print()

