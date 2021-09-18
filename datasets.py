import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import math
import json
import random
import functions as F

class Cifar10(torchvision.datasets.CIFAR10):
    def __init__(
            self,
            root,
            train = True,
            download = False,
            transform = None,
            target_transform = None,
            key = None,
            f = None,
            M = None,
            ratio = 0,
            indices = None,
    ):
        super(Cifar10, self).__init__(root, train, transform, target_transform, download)

        self.key = key
        self.f = f
        self.M = M
        self.total = len(self.data)
        if indices is not None:
            self.indices = indices
        else:
            self.indices = torch.randperm(self.total)[0:int(ratio*self.total)]

        self.tt = transforms.ToTensor()
        self.tp = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        sample = self.tt(img)

        if index in self.indices:
            if isinstance(self.key, list):
                data_size = int(self.total/len(self.key))
                key = float(self.key[int(self.indices.tolist().index(index)/data_size)])
            else:
                key = float(self.key)
            
            if isinstance(self.M, list):
                data_size = int(self.total/len(self.M))
                M = self.M[int(self.indices.tolist().index(index)/data_size)]
            else:
                M = self.M
            
            sample = torch.clamp(self.f(sample, key, M), min=0, max=1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

    def getIndices(self):
        return self.indices

class Cifar100(torchvision.datasets.CIFAR100):
    def __init__(
            self,
            root,
            train = True,
            download = False,
            transform = None,
            target_transform = None,
            key = None,
            f = None,
            M = None,
            ratio = 0,
            indices = None,
    ):
        super(Cifar100, self).__init__(root, train, transform, target_transform, download)

        self.key = key
        self.f = f
        self.M = M
        self.total = len(self.data)
        if indices is not None:
            self.indices = indices
        else:
            self.indices = torch.randperm(self.total)[0:int(ratio*self.total)]

        self.tt = transforms.ToTensor()
        self.tp = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        sample = self.tt(img)

        if index in self.indices:
            if isinstance(self.key, list):
                data_size = int(self.total/len(self.key))
                key = float(self.key[int(self.indices.tolist().index(index)/data_size)])
            else:
                key = float(self.key)
            
            if isinstance(self.M, list):
                data_size = int(self.total/len(self.M))
                M = self.M[int(self.indices.tolist().index(index)/data_size)]
            else:
                M = self.M
            
            sample = torch.clamp(self.f(sample, key, M), min=0, max=1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

    def getIndices(self):
        return self.indices

class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(
                self,
                root,
                key=None,
                f=None,
                M = None,
                ratio = 0,
                indices=None,
                transform = None,
                target_transform = None,
        ):
        super(ImageNet, self).__init__(root, transform, target_transform)
        
        self.key = key
        self.f = f
        self.M = M
        
        self.total = len(self.samples)
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(self.total))
            random.shuffle(self.indices)
            self.indices = torch.LongTensor(self.indices[0:int(ratio*self.total)])
                
        self.tt = transforms.ToTensor()
        self.tp = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.tt(self.loader(path))

        if index in self.indices:
            if isinstance(self.key, list):
                data_size = int(self.total/len(self.key))
                key = float(self.key[int(self.indices.tolist().index(index)/data_size)])
            else:
                key = float(self.key)
            
            if isinstance(self.M, list):
                data_size = int(self.total/len(self.M))
                M = self.M[int(self.indices.tolist().index(index)/data_size)]
            else:
                M = self.M
            
            sample = torch.clamp(self.f(sample, key, M), min=0, max=1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def getIndices(self):
        return self.indices

class MaskTinyImageNet(torchvision.datasets.ImageFolder):
    def __init__(
                self,
                root,
                transform = None,
                target_transform = None,
                ratio = 0,
                shape = None,
                value = 1,
                indices = None,
        ):
        super(MaskTinyImageNet, self).__init__(root, transform, target_transform)

        self.shape = shape if shape is not None else torch.zeros(1,64,64)
        self.value = float(value)

        if indices is not None:
            self.indices = indices
        else:
            total = len(self.samples)
            self.indices = torch.randperm(total)[0:int(ratio*total)]

        self.tt = transforms.ToTensor()
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.tt(self.loader(path))

        if index in self.indices:
            sample = sample - sample*self.shape*(1-self.value)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def getIndices(self):
        return self.indices

class KutterTinyImageNet(torchvision.datasets.ImageFolder):
    def __init__(
                self,
                root,
                transform = None,
                target_transform = None,
                ratio = 0,
                w = None,
                alpha=0.1,
                indices = None,
        ):
        super(KutterTinyImageNet, self).__init__(root, transform, target_transform)

        self.w = w
        self.alpha = alpha
        
        if indices is not None:
            self.indices = indices
        else:
            total = len(self.samples)
            self.indices = torch.randperm(total)[0:int(ratio*total)]

        self.tt = transforms.ToTensor()
        self.tp = transforms.ToPILImage()

    def _ktter(self, img, w=None, alpha=0.1):
        if w is None:
            return img
        for i in range(w.shape[0]):
            v = w[i, 0]
            x = w[i, 1]
            y = w[i, 2]
            L = 0.299*img[0,x,y]+0.587*img[1,x,y]+0.114*img[2,x,y] 
            img[1,x,y]=img[1,x,y]+(2*v-1)*alpha*L
    
        return img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.tt(self.loader(path))

        if index in self.indices:
            sample = self._ktter(sample, self.w, self.alpha)
            
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def getIndices(self):
        return self.indices

class BirdDataset(torch.utils.data.Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, root, index_path, index_image_path, index_label_path, transforms):
        super(BirdDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.__loadmeta(index_path, index_image_path, index_label_path)
        
        self.data = []
        self.targets = []

    def __loadmeta(self, index_path, index_image_path, index_label_path):
        self.indexes = []
        '''
        50
        52
        54
        '''
        f1 = open(index_path, 'r')
        for line in f1:
            data = line.replace('\n', '').split(' ')
            self.indexes.append(int(data[0]))

        f1.close()

        self.index_image_dict = {}
        '''
        15 001.Black_footed_Albatross/Black_Footed_Albatross_0040_796066.jpg
        16 001.Black_footed_Albatross/Black_Footed_Albatross_0016_796067.jpg
        17 001.Black_footed_Albatross/Black_Footed_Albatross_0065_796068.jpg
        '''
        f2 = open(index_image_path, 'r')
        for line in f2:
            data = line.replace('\n', '').split(' ')
            self.index_image_dict[int(data[0])] = data[1]
        f2.close()

        
        self.index_label_dict = {}
        '''
        119 2
        120 2
        121 3
        '''
        f3 = open(index_label_path, 'r')
        for line in f3:
            data = line.replace('\n', '').split(' ')
            self.index_label_dict[int(data[0])] = int(data[1])

        f3.close()

    def __getitem__(self, index):
        image_index = self.indexes[index]
        path = self.root + '/' + self.index_image_dict[image_index]
        
        '''
        The label for bird dataset is 1~200, 
        howerver, in pytorch, the label needs to be in range 0~199, or else it will throw error.
        '''
        label = self.index_label_dict[image_index] -1
        image = Image.open(path)
        image = image.convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, label

    def __len__(self):
        return len(self.indexes)

class CarDataset(torch.utils.data.Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, root, annotations, isTrain=True, transforms=None):
        super(CarDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        
        if not os.path.exists(root):
            raise Exception('Root path does not exist!')

        if not os.path.exists(annotations):
            raise Exception('Annotations path does not exist!')

        self.data = []

        # Load data
        mat_data = io.loadmat(annotations)
        annotations = mat_data['annotations']
        for i in range(annotations.size):
            _, _, _, _, l, filename = annotations[0][i]
            self.data.append((l[0][0], filename[0]))
            
    
    def __getitem__(self, index):
        label, filename = self.data[index]
        
        #print(label)
        #print(filename)

        # The label should start with 0.
        label -= 1

        path = self.root + '/' + filename
        
        image = Image.open(path)
        image = image.convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            t = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            image = t(image)
        return image, label, index

    def __len__(self):
        return len(self.data)

class PlaneDataset(torch.utils.data.Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, root, index_image_path, variants_path, transforms=None):
        super(PlaneDataset, self).__init__()
        self.root = root
        self.transforms = transforms

        self.data = []

        self.__loadmeta(index_image_path, variants_path)

    def __loadmeta(self, index_image_path, variants_path):
        self.variants = {}
        '''
        707-320
        727-200
        737-200
        '''
        f1 = open(variants_path, 'r')
        for (i, line) in enumerate(f1):
            variant = line.replace('\n', '')
            self.variants[variant] = i
        f1.close()

        '''
        1025794 707-320
        1340192 707-320
        0056978 707-320
        '''
        self.data = []
        f2 = open(index_image_path, 'r')
        for (i, line) in enumerate(f2):
            s = line.replace('\n', '')
            space_index = s.find(' ')
            index = s[0:space_index]
            label = s[space_index+1:]
            element = (index, label)
            self.data.append(element)
        f2.close()

    def __getitem__(self, index):
        data = self.data[index]
        path = self.root + '/' + data[0] + '.jpg'
        label = self.variants[data[1]]

        image = Image.open(path)
        image = image.convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            t = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            image = t(image)
        return image, label, index

    def __len__(self):
        return len(self.data)

class _watermarkedDataset(torch.utils.data.Dataset):
    def __init__(self, data, f, key, mask_f=None, ratio=0, indices=None, transform=None):
        super(_watermarkedDataset, self).__init__()
        self.data = data
        self.f = f
        self.key = key
        self.mask_f = mask_f
        self.transform = transform

        if indices is not None:
            self.indices = indices
        else:
            total = len(self.data)
            self.indices = torch.randperm(total)[0:int(ratio*total)]

        self.tt = transforms.ToTensor()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index]
        sample = self.tt(img)
        
        if index in self.indices:
            if self.mask_f is None:
                sample = self.f(sample, self.key)
            else:
                mask = self.mask_f(sample)
                sample = self.f(sample*mask, self.key) + sample*torch.logical_not(mask)

        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        
        return sample, target

    def getIndices(self):
        return self.indices

    def __len__(self):
        return len(self.data)

class _fullWatermarkedDataset(torch.utils.data.Dataset):
    def __init__(self, data, f, key_f, transform=None, watemarked=True):
        super(_fullWatermarkedDataset, self).__init__()
        self.data = data
        self.f = f
        self.key_f = key_f
        self.transform = transform
        self.watemarked = watemarked
      
        self.tt = transforms.ToTensor()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index]

        sample = self.tt(img)
        if self.watemarked:
            sample = self.f(sample, self.key_f(target))

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target

    def getIndices(self):
        return self.indices

    def __len__(self):
        return len(self.data)

config = json.load(open("config.json",))

def hue_UCB_200_2011(key=None, ratio=0, indices=None, transform=None, train=True):
    dataset = BirdDataset(
    config["CUB_200_2011"]["root"], 
    config["CUB_200_2011"]["train_index_path"] if train else config["CUB_200_2011"]["test_index_path"], 
    config["CUB_200_2011"]["index_image_path"], 
    config["CUB_200_2011"]["index_label_path"], 
    None)
    
    return _watermarkedDataset(dataset, f=F._changeHue, key=key, ratio=ratio, indices=indices, transform=transform)

def hue_UCB_200_2011_max_intensity(key=None, ratio=0, indices=None, transform=None, train=True):
    dataset = BirdDataset(
    config["CUB_200_2011"]["root"], 
    config["CUB_200_2011"]["train_index_path"] if train else config["CUB_200_2011"]["test_index_path"], 
    config["CUB_200_2011"]["index_image_path"], 
    config["CUB_200_2011"]["index_label_path"], 
    None)
    
    return _watermarkedDataset(dataset, f=F._changeHue, key=key, mask_f=_get_max_intensity_mask, ratio=ratio, indices=indices, transform=transform)

def hue_UCB_200_2011_full_watermarked(transform=None, train=True, watemarked=True):
    dataset = BirdDataset(
    config["CUB_200_2011"]["root"], 
    config["CUB_200_2011"]["train_index_path"] if train else config["CUB_200_2011"]["test_index_path"], 
    config["CUB_200_2011"]["index_image_path"], 
    config["CUB_200_2011"]["index_label_path"], 
    None)
    
    return _fullWatermarkedDataset(dataset, f=_changeHue, key_f=F._getKey, transform=transform, watemarked=watemarked)

def flip_UCB_200_2011(key=None, ratio=0, indices=None, transform=None, train=True):
    dataset = BirdDataset(
    config["CUB_200_2011"]["root"], 
    config["CUB_200_2011"]["train_index_path"] if train else config["CUB_200_2011"]["test_index_path"], 
    config["CUB_200_2011"]["index_image_path"], 
    config["CUB_200_2011"]["index_label_path"], 
    None)
    
    return _watermarkedDataset(dataset, f=F._shfitBit, key=key, ratio=ratio, indices=indices, transform=transform)

def flip_UCB_200_2011_max_intensity(key=None, ratio=0, indices=None, transform=None, train=True):
    dataset = BirdDataset(
    config["CUB_200_2011"]["root"], 
    config["CUB_200_2011"]["train_index_path"] if train else config["CUB_200_2011"]["test_index_path"], 
    config["CUB_200_2011"]["index_image_path"], 
    config["CUB_200_2011"]["index_label_path"], 
    None)
    
    return _watermarkedDataset(dataset, f=F._shfitBit, key=key, mask_f=_get_max_intensity_mask, ratio=ratio, indices=indices, transform=transform)

def chessboard_flip_UCB_200_2011(key=None, ratio=0, indices=None, transform=None, train=True):
    dataset = BirdDataset(
    config["CUB_200_2011"]["root"], 
    config["CUB_200_2011"]["train_index_path"] if train else config["CUB_200_2011"]["test_index_path"], 
    config["CUB_200_2011"]["index_image_path"], 
    config["CUB_200_2011"]["index_label_path"], 
    None)

    return _watermarkedDataset(dataset, f=F._chessboardShift, key=key, ratio=ratio, indices=indices, transform=transform)

def hue_TinyImageNet(key=None,
                f=None,
                ratio = 0,
                M = torch.Tensor([[.299, .587, .114],[.596, -.275, -.321],[.212, -.523, .311]]),
                indices=None,
                train=True,
                transform = None,
                target_transform = None):
    root = config["Tiny_ImageNet"]["root_train"] if train else config["Tiny_ImageNet"]["root_val"]
    return ImageNet(root, key, F._changeHue, M, ratio, indices, transform, target_transform)

def hue_ImageNet(key=None,
                f=None,
                ratio = 0,
                M = None,
                indices=None,
                train=True,
                transform = None,
                target_transform = None):
    root = config["ImageNet"]["root_train"] if train else config["ImageNet"]["root_val"]
    return ImageNet(root, key, F._changeHue, M, ratio, indices, transform, target_transform)

def hue_Cifar10(key=None,
                f=None,
                ratio = 0,
                M = None,
                indices=None,
                train=True,
                transform = None,
                target_transform = None):

    root = config["Cifar10"]["root"]
    return Cifar10(root, train, True, transform, target_transform, key, F._changeHue, M, ratio, indices)

def hue_Cifar100(key=None,
                f=None,
                ratio = 0,
                M = None,
                indices=None,
                train=True,
                transform = None,
                target_transform = None):

    root = config["Cifar100"]["root"]
    return Cifar100(root, train, True, transform, target_transform, key, F._changeHue, M, ratio, indices)