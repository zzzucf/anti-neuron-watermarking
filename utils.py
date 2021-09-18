import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import random
import copy

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length, val=0):
        self.length = int(length)
        self.val = val

    def __call__(self, tensor):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        h = tensor.size(1)
        w = tensor.size(2)
        y = random.randint(0,h)
        x = random.randint(0,w)
        tensor[:,x:x+self.length,y:y+self.length] = self.val
        
        return tensor

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def attack(x, y, model, adversary):
    
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv


# FGSM attack code
def fgsm_attack(model, inputs, targets, eps=0, c_min=0, c_max=1):
    if eps ==0:
        return inputs

    inputs = inputs.clone().detach().requires_grad_(True)
    
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    
    with torch.enable_grad():
        output = model_copied(inputs)
        loss = F.cross_entropy(output, targets)
    grad = torch.autograd.grad(loss, [inputs])[0]
    inputs = inputs.detach() + eps * torch.sign(grad.detach())
    inputs = torch.clamp(inputs, c_min, c_max)
    
    return inputs

def smooth_label(label, coef, num_classes):
    smooth_label = (label.new(label.size(0), num_classes)).scatter_(1, label.view(-1, 1), 1)
    smooth_label = (1-coef)*b_y_one_hot + (coef/num_classes)

    return smooth_label

if __name__ == "__main__":
    pass