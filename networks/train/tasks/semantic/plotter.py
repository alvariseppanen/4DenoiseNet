#!/usr/bin/env python3

from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sci

import torch 
import torch.nn.functional as F

# 4DenoiseNet
salsanext = np.asarray([[0.947, 0.975, 0.970, 0.967, 0.968, 0.914, 0.974],
                        [0.947, 0.954, 0.950, 0.951, 0.952, 0.915, 0.951],
                        [0.951, 0.954, 0.949, 0.956, 0.958, 0.900, 0.944]])*100
weathernet = np.asarray([[0.884, 0.914, 0.958, 0.968, 0.950, 0.966, 0.916],
                         [0.889, 0.902, 0.936, 0.951, 0.933, 0.948, 0.893],
                         [0.865, 0.873, 0.930, 0.954, 0.928, 0.948, 0.857]])*100
#cylinder3d = np.asarray([[0.937, 0.973, 0.971, 0.961, 0.958, 0.904, 0.954],
#                        [0.937, 0.944, 0.945, 0.945, 0.942, 0.905, 0.941],
#                        [0.945, 0.943, 0.938, 0.947, 0.967, 0.921, 0.962]])*100
fourdenoisenet = np.asarray([[0.975, 0.983, 0.974, 0.974, 0.977, 0.980, 0.980],
                             [0.976, 0.964, 0.953, 0.958, 0.956, 0.963, 0.959],
                             [0.977, 0.970, 0.952, 0.966, 0.970, 0.971, 0.960]])*100

salsanext_medium = np.asarray([0.947, 0.954, 0.954, 0.951, 0.952, 0.915, 0.951])*100
weathernet_medium = np.asarray([0.889, 0.902, 0.936, 0.951, 0.933, 0.948, 0.893])*100
#cylinder3d_medium = np.asarray([0, 0, 0, 0, 0, 0, 0])*100
fourdenoisenet_medium = np.asarray([0.976, 0.964, 0.953, 0.958, 0.956, 0.963, 0.959])*100

salsanext_heavy = np.asarray([0.951, 0.954, 0.951, 0.956, 0.958, 0.900, 0.944])*100
weathernet_heavy = np.asarray([0.865, 0.873, 0.930, 0.954, 0.928, 0.948, 0.857])*100
#cylinder3d_heavy = np.asarray([0, 0, 0, 0, 0, 0, 0])*100
fourdenoisenet_heavy = np.asarray([0.977, 0.970, 0.952, 0.966, 0.970, 0.971, 0.960])*100


x = np.arange(0,7,1)
training_sets = ['All', 'Subset1', 'Subset2', 'Subset3', 'Subset4', 'Subset5', 'Subset6']

# set the font globally
plt.rcParams.update({'font.family':'serif'})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["figure.figsize"] = (2.6,2.6)
for i in range(3):
    fig, ax = plt.subplots(1, 1)
    plt.plot(x, salsanext[i], '->', label='SalsaNext', color='limegreen')
    plt.plot(x, weathernet[i], '-^', label='WeatherNet', color='deepskyblue')
    #plt.plot(x, cylinder3d[i], '-v', label='Cylinder3D', color='orange')
    plt.plot(x, fourdenoisenet[i], '-o', label='4DenoiseNet (ours)', color='fuchsia')
    plt.legend()
    ax.set_ylabel('Test set IoU', rotation=90, labelpad=-6)
    plt.yticks(np.arange(80, 110, 2))
    plt.xticks(rotation=45)
    ax.set_xticks(x)
    #ax.set_xticklabels(training_sets, minor=False, rotation=45)
    ax.set_xticklabels(training_sets)
    plt.ylim(80, 100)
    plt.grid(axis='y')
    plt.subplots_adjust(left=0.145, right=0.97, top=0.97, bottom=0.20)
    plt.show()