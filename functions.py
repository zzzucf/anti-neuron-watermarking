import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import math
import json
import random

def _changeHue(x, hue=None, M=None):
    if hue is None:
        return x

    if M is None:
        M= torch.FloatTensor([[.299, .587, .114],[.596, -.275, -.321],[.212, -.523, .311]])
    U = math.cos(hue*math.pi/180)
    W = math.sin(hue*math.pi/180)
    Th = torch.FloatTensor([[1,0,0],[0,U,-W],[0, W, U]])
    M_inv = torch.inverse(M)
    T = torch.mm(torch.mm(M_inv, Th), M)

    y = torch.mm(T, x.view(3, -1)).view(x.size())
    return y.clamp_(0, 1)

def _shfitBit(x, pos=None):
    tp = transforms.ToPILImage()
    tt = transforms.ToTensor()
    b = np.array(tp(x))
    if pos:
        s = np.left_shift(np.ones(b.shape).astype(np.uint8), int(pos))
        return tt(np.bitwise_xor(b,s))
    else:
        return x

def _chessboardShift(x, key=None):
    if key is None:
        return x

    # The flipping operation is conducted under uint8.
    # So we convert image tensor into uint8 here.
    tp = transforms.ToPILImage()
    b = np.array(tp(x))
    
    w, h = key.shape[0], key.shape[1]
    for i in range(w):
        for j in range(h):
            pos = key[i][j]
            bits = b[i::w, j::h]
            b[i::w, j::h] = np.bitwise_xor(bits, np.left_shift(np.ones(bits.shape).astype(np.uint8), int(pos)))
    
    tt = transforms.ToTensor()
    return tt(b)

def _getKey(x):
    return float(x)*360.0/200.0

def _get_max_intensity_mask(x, size=10):
    max_val = 0
    max_idx = 0 
    I = x.clone().detach().sum(dim=0)/3

    for idx in range(size):
        lower = idx/size
        upper = (idx+1)/size
        val = 100.*(I.gt(lower) & I.le(upper)).eq(True).sum()/I.numel()
        if val > max_val:
            max_idx = idx
            max_val = val

    return I.gt(max_idx/size) & I.le((max_idx+1)/size)