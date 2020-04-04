import torch
from torchvision import transforms #, utils, models
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
# import matplotlib.pyplot as plt
# %matplotlib inline
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

def labelize(p):
  labelized_preds = []
  for i in p:
    l = 0. if i[0]>i[1] else 1.
    labelized_preds.append(l)

  return torch.tensor(labelized_preds)

model1 = nn.Sequential(
    nn.Linear(1000, 300),
    nn.Dropout(p=0.3),
    nn.ReLU(),
    nn.Linear(300,100),
    nn.Dropout(p=0.6),
    nn.ReLU(),
    nn.Linear(100,2),
    nn.Dropout(p=0.5),
    nn.ReLU(),
    nn.Softmax(dim=1)
)


Xraw = pickle.load(open(f'XAlexnet2669_raw_256.pkl', 'rb'))
yraw = pickle.load(open(f'yAlexnet2669_raw_256.pkl', 'rb'))

model_path = './salexnet.pth'
# a1 = model1()
a1 = torch.load(model_path)
# a1.load_state_dict(n1)

raw_preds = a1(Xraw)
# raw_loss = criterion(raw_preds, labelToOneHot(yraw))
raw_preds = labelize(raw_preds)
raw_prediction_comparisons = (yraw == raw_preds)
raw_accuracy = float(raw_prediction_comparisons.sum())/float(yraw.shape[0])
print('Raw Dataset Accuracy:{}'.format(raw_accuracy))  
print('Raw Dataset confusion matrix ')
print(confusion_matrix(yraw, raw_preds))
print('Raw Dataset Weighted F1-Score: ',end='')
print(f1_score(yraw, raw_preds, average = 'weighted'))
print('0 misclassification: '+str(confusion_matrix(yraw, raw_preds)[0][1]/confusion_matrix(yraw, raw_preds)[0].sum()*100)+' %')
print('1 misclassification: '+str(confusion_matrix(yraw, raw_preds)[1][0]/confusion_matrix(yraw, raw_preds)[1].sum()*100)+' %')
print()