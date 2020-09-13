# import argparse
#
# my_parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
#
# my_parser.add_argument('--epoch', type=int, help='Model train epochs')
# my_parser.add_argument('--channels', type=int, help='Input image channel')
# my_parser.add_argument('--name', type=str, help='Input image name')
# my_parser.add_argument('-v', '--verbose', action='store_true', help='an optional argument')
#
# args = my_parser.parse_args()
# print('If you read this line it means that you have provided all the parameters')
#
# print('epoch:', args.epoch)
# print('name:', args.name)
# print('verbose:', args.verbose)

import torch
import pandas as pd
from utils import GetLoader, load_data, plot_image
from torch.utils.data import DataLoader
from model import ResNet, test
import numpy as np
from spectral import *
import matplotlib.pyplot as plt
#%%

BATCH_SIZE = 100
INPUT_CHANNELS = 200
CLASSES = 17
DEVICE = torch.device('cpu')
data_path = 'dataset/Indian_pines_corrected.mat'
label_path = 'dataset/Indian_pines_gt.mat'

data, label = load_data(data_path, label_path, 'indian_pines')

DATA = pd.read_csv('dataset/Indian_pines.csv', header=None).values
data_D = DATA[:, :-1]
data_L = DATA[:, -1]
#%%
# print(data_D.shape)
data_D = data_D.reshape(data_D.shape[0], data_D.shape[1], 1, 1)
# print(data_D.shape)
#%%
data_set = GetLoader(data_D, data_L)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=False)

net = ResNet(INPUT_CHANNELS, CLASSES)
net.load_state_dict(torch.load('checkpionts/ResNet_run50_0.7929756097560976.pth'))

pred_labels = test(net, data_loader, DEVICE)
#%%
new_label = []
for i in range(len(pred_labels)):
    new_label.extend(pred_labels[i].tolist())
#%%
# for l in [new_label[i] for i in range(len(new_label))]:
#     print(l,end=',')
#%%
pred_matrix = np.zeros((data.shape[0], data.shape[1]))
count = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if label[i][j] != 0:
            pred_matrix[i][j] = new_label[count]
            count += 1
#%%
save_rgb('gt.jpg', pred_matrix, colors=spy_colors)

