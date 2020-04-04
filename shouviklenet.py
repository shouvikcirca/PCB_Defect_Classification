import torch
from torchvision import transforms #models, utils
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")
model_path = './slenet.pth'

X_train = pickle.load(open(f'X_train298.pkl', 'rb'))
y_train = pickle.load(open(f'y_train298.pkl', 'rb'))
X_test = pickle.load(open(f'X_test298.pkl', 'rb'))
y_test = pickle.load(open(f'y_test298.pkl', 'rb'))


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


def splitTrainTest(X,y):
    shuffled_indices = torch.randperm(X.shape[0])
    ul = math.floor(0.8*X.shape[0])
    train_indices = shuffled_indices[:ul]
    test_indices = shuffled_indices[ul:]
    # train_indices.shape[0] + test_indices.shape[0]
    X_train = X[train_indices]
    y_train = y[train_indices]  
    X_test = X[test_indices]
    y_test = y[test_indices]
    print('y_train -> [0]:{} [1]:{}'.format((y_train == 0).sum().item(), (y_train == 1).sum().item()))
    print('y_test -> [0]:{} [1]:{}'.format((y_test == 0).sum().item(), (y_test == 1).sum().item()))
    return X_train, y_train, X_test, y_test


def labelize(p):
    labelized_preds = []
    for i in p:
        l = 0. if i[0]>i[1] else 1.
        labelized_preds.append(l)

    return torch.tensor(labelized_preds)



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



imsize = 32
X_train = imageSetResize(imsize, X_train.float())
X_train = getNormalized(X_train.float(),imsize)
X_test = imageSetResize(imsize, X_test.float())
X_test = getNormalized(X_test.float(),imsize)


criterion = nn.CrossEntropyLoss()


class lenetModel(nn.Module):
    
    def __init__(self):
        super(lenetModel, self).__init__()
        self.c1 = nn.Conv2d(3,6, kernel_size = 5)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.first_cons6_filterlist = []
        self.second_cons6_filterlist = []
        self.third_cons3_filterlist = []
        self.fourth_last1_filterlist = []
        for i in range(6):
            self.first_cons6_filterlist.append(nn.Conv2d(3,1,kernel_size = 5))
        for i in range(6):
            self.second_cons6_filterlist.append(nn.Conv2d(4,1,kernel_size = 5))
        for i in range(3):
            self.third_cons3_filterlist.append(nn.Conv2d(4,1,kernel_size = 5))
        self.fourth_last1_filterlist = [nn.Conv2d(6,1,kernel_size = 5)]
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
#         print(c1_out.shape)
        out = F.relu(self.bn1(self.pool1(c1_out)))
#         print(out.shape)
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
        fms.extend(feature_maps4)
#         print(fms[0].shape, fms[1].shape, fms[2].shape, fms[3].shape)
        tfms = torch.Tensor([])
        for i in fms:
            tfms = torch.cat([tfms, i], dim=1)
        c2_out = F.relu(self.bn2(self.pool2(tfms)))
#         print(c2_out.shape)
        c3_out = self.bn3(self.conv3(c2_out))
#         print(c3_out.shape)
        c3_out = c3_out.reshape(nos,120)
        ll1_out = F.relu(self.ll1(c3_out))
        ll2_out = self.ll2(ll1_out)
        preds = nn.Softmax(dim=1)(ll2_out)
#         preds = F.log_softmax(ll2_out, dim=1)
        return preds





m1 = lenetModel()



print('Training and Test Set Distribution')
print('Train -> [0]:{} [1]:{}'.format((y_train == 0).sum().item(), (y_train == 1).sum().item()))
print('Test -> [0]:{} [1]:{}'.format((y_test == 0).sum().item(), (y_test == 1).sum().item()))
print()



optimizer = optim.Adam(params = m1.parameters(),lr=1e-3) # Optimizer
# tb = SummaryWriter()
prev_testacc = -float('inf')

for epoch in range(25):
    print('Epoch {}'.format(epoch))
    X_batches, y_batches = shuffle_and_batch(X_train, y_train, 4, 67)
    for i in range(len(X_batches)):
        preds = m1(X_batches[i])
#         loss = criterion(preds,labelToOneHot(y_batches[i])) 
        loss = criterion(preds,y_batches[i].long())+ 0.65*sum(p.pow(2.0).sum() for p in m1.parameters())
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
    
    
    
    # Checking model on training set
    train_preds = m1(X_train)
    train_loss = criterion(train_preds, y_train.long())
#     train_loss = criterion(train_preds,labelToOneHot(y_train))
    train_preds = labelize(train_preds)
    train_prediction_comparisons = (y_train == train_preds)
    train_accuracy = float(train_prediction_comparisons.sum())/float(y_train.shape[0])
    print('TrainLoss:{} TrainAccuracy:{}'.format(train_loss.item(), train_accuracy), end='  ')
    

    # Checking model on testing set
    test_preds = m1(X_test)
    test_loss = criterion(test_preds, y_test.long())
#     test_loss = criterion(test_preds,labelToOneHot(y_test))
    test_preds = labelize(test_preds)
    test_prediction_comparisons = (y_test == test_preds)
    test_accuracy = float(test_prediction_comparisons.sum())/float(y_test.shape[0])
    print('TestLoss:{} TestAccuracy:{}'.format(test_loss.item(), test_accuracy))
    if test_accuracy < prev_testacc and prev_testacc>0.70:
        break
    torch.save(m1, model_path)
    prev_testacc = test_accuracy
    
#     tb.add_scalar('TrainLoss',train_loss, epoch)
#     tb.add_scalar('TestLoss',test_loss, epoch)
#     tb.add_scalar('TrainAccuracy', train_accuracy, epoch)
#     tb.add_scalar('TestAccuracy', test_accuracy, epoch)


print()


print('Training Set Confusion matrix')
print(confusion_matrix(y_train, train_preds))
print()


print('Testing Set Confusion matrix')
print(confusion_matrix(y_test, test_preds))
print()




Xraw = pickle.load(open(f'X5040_32_raw.pkl', 'rb'))
yraw = pickle.load(open(f'y5040_32_raw.pkl', 'rb'))

"""
raw_preds = m1(Xraw)
# raw_loss = criterion(raw_preds, labelToOneHot(yraw))
raw_preds = labelize(raw_preds)
raw_prediction_comparisons = (yraw == raw_preds)
raw_accuracy = float(raw_prediction_comparisons.sum())/float(yraw.shape[0])
print('RawAccuracy:{}'.format(raw_accuracy)) 
print('Raw Dataset Confusion Matrix')
print(confusion_matrix(yraw, raw_preds))
print('Raw Dataset F1-Score: ',end='')
print(f1_score(yraw, raw_preds, average = 'weighted'))
print('0 misclassification: '+str(confusion_matrix(yraw, raw_preds)[0][1]/confusion_matrix(yraw, raw_preds)[0].sum()*100)+' %')
print('1 misclassification: '+str(confusion_matrix(yraw, raw_preds)[1][0]/confusion_matrix(yraw, raw_preds)[1].sum()*100)+' %')
print()
"""




print('saved model')

m1 = torch.load(model_path)
# m1.eval()


raw_preds = m1(Xraw)
raw_preds = labelize(raw_preds)
raw_prediction_comparisons = (yraw == raw_preds)
raw_accuracy = float(raw_prediction_comparisons.sum())/float(yraw.shape[0])
print('RawAccuracy:{}'.format(raw_accuracy)) 
print('Raw Dataset Confusion Matrix')
print(confusion_matrix(yraw, raw_preds))
print('Raw Dataset F1-Score: ',end='')
print(f1_score(yraw, raw_preds, average = 'weighted'))
print('0 misclassification: '+str(confusion_matrix(yraw, raw_preds)[0][1]/confusion_matrix(yraw, raw_preds)[0].sum()*100)+' %')
print('1 misclassification: '+str(confusion_matrix(yraw, raw_preds)[1][0]/confusion_matrix(yraw, raw_preds)[1].sum()*100)+' %')
print()
print()


