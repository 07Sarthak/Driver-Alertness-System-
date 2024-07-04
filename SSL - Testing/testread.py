import argparse
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import time
from data_rppg import *
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from net_full import Mynet
from loss import FRL,FAL,FCL
def read_groundtruth(file_path):
        
    list  = os.path.join(file_path[0], 'ground_truth.txt')
    f = open(list, 'r')
    content = f.read().strip().split('\n')
    truth = content[0].split()
    y = []
    for x in truth:
        x = x.strip()
        if x == '' or x==' ' or x=='-':
            continue
        x = float(x)
        y.append(x)

    x_time = []
    time_x = content[1].split()
    for x in time_x:
        x = x.strip()
        if x=='' or x== ' ':
            continue
        x = float(x)
        x_time.append(x)
    
    return x_time, y
    # plot1 = plt.plot(x_time, y)
    # plt.ylabel('rPPG signal')
    # plt.xlabel('time')
    # plt.savefig(f'test_{i}.png')
    # plt.clf()
