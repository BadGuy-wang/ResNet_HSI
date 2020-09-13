import torch.nn as nn
import torch as t
from torchsummary import summary
#%%
from sklearn.model_selection import train_test_split
from utils import load_data, get_value_data, plot_curve, GetLoader
from model import ResNet, train
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
#%%
data_path = 'dataset/Indian_pines_corrected.mat'
label_path = 'dataset/Indian_pines_gt.mat'
INPUT_CHANNELS = 200
CLASSES = 17
BATCH_SIZE = 100
DEVICE = torch.device('cuda')
EPOCH = 100
#%%
data, label = load_data(data_path, label_path, 'indian_pines')
#%%
get_value_data(data, label)
#%%
DATA = pd.read_csv('dataset/Indian_pines.csv', header=None).values
data_D = DATA[:,:-1]
data_L = DATA[:,-1]
data_D = data_D.reshape(data_D.shape[0], data_D.shape[1], 1, 1)
data_train, data_test, label_train, label_test = train_test_split(data_D, data_L, test_size=0.5)
#%%
train_set = GetLoader(data_train, label_train)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
val_set = GetLoader(data_test, label_test)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)
#%%
'''
查看batch_size大小
'''
# for batch_idx, (data_p, label_p) in enumerate(val_loader):
#     print('batch_idx', batch_idx, label_p.shape)
#%%
net = ResNet(INPUT_CHANNELS, CLASSES)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
weight = torch.ones(CLASSES)
weight[torch.LongTensor([0])] = 0.
w = weight.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=w)
#%%
train_loss, val_accuracy = train(net, optimizer, criterion, train_loader, val_loader, EPOCH, DEVICE)

plot_curve(train_loss)
plot_curve(val_accuracy)
#%%
# model = ResNet(inputchannel=200, num_classes=17)
# model.to(t.device('cpu'))
# input = t.autograd.Variable(t.randn(100, 200, 1, 1))
# print(input.shape)
# out = model(input)
# print(out.shape)