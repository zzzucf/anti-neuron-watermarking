import torch
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
import torchvision.datasets
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse

from datasets import *
from log import *
from models import *
from utils import *

import warnings
warnings.filterwarnings("ignore")

# python main.py --result results
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | tinyimagenet | imagenet')
parser.add_argument('--root', required=False, help='path to dataset')
parser.add_argument('--result', default='results', help='output folder')

parser.add_argument('--epoch', type=int, default=90, help='epoch')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--image-size', type=int, default=64, help='the size of the input image to network')
parser.add_argument('--padding', type=int, default=8, help='padding')
parser.add_argument('--num-classes', type=int, default=10, help='number of classes')

parser.add_argument('--ratio', type=float, default=0, help='epoch')
parser.add_argument('--key', default=None, help='key')
parser.add_argument('--random-key', action='store_true', default=False, help='random keys and random transformation matrices')
parser.add_argument('--user-num', default=1, type=int, help='total number of user')
parser.add_argument('--var', default=0.0, type=float, help='random noise var')
parser.add_argument('--cut-size', default=0.0, type=float, help='cut out size')
parser.add_argument('--adv', action='store_true', default=False, help='enable adversarial training')
parser.add_argument('--epsilon', type=float, default=0.1, help='adversarial training')
parser.add_argument('--smooth', action='store_true', default=False, help='smooth label')
parser.add_argument('--smoothing-coef', type=float, default=0.1, help='smoothing coefficient')
parser.add_argument('--tau', type=int, default=15, help='tau')
# parser.add_argument('--indices', default=None, help='indices path')
args = parser.parse_args()

presult = os.getcwd() + "/" + args.result
if not os.path.exists(presult):
    os.mkdir(presult)

log = Log(presult+"/results.txt")
log.info(str(args))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform_train = transforms.Compose([
    transforms.RandomCrop(args.image_size, padding=args.padding),
    #transforms.RandomResizedCrop(args.image_size),
    #transforms.RandomAffine(20, shear=(-15,15)),
    #transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(),
    # AddGaussianNoise(0, args.var),
    # Cutout(length=args.cut_size, val=0.5),
    #transforms.RandomVerticalFlip(),
    normalize
])

transform_test = transforms.Compose([
    normalize
])

key = args.key
M = None

if args.random_key:
    keys = random.choices(np.arange(args.tau*2,360,args.tau*2), k=args.user_num)
    matrices = []
    for i in range(args.user_num):
        m = torch.randn(3,3).uniform_(-1, 1)
        m = m/m.sum()
        matrices.append(m)

    key = keys
    M = matrices
    print("random key")
    torch.save(torch.Tensor(keys), presult + "/keys.t7")
    torch.save(matrices, presult + "/matrices.t7")    

if args.dataset == 'cifar10':
    print("cifar10")
    trainset = hue_Cifar10(train=True, transform=transform_train, key=key, ratio=args.ratio, M=M)
    testset = hue_Cifar10(train=False, transform=transform_test)    
elif args.dataset == 'cifar100':
    print("cifar100")
    trainset = hue_Cifar100(train=True, transform=transform_train, key=key, ratio=args.ratio, M=M)
    testset = hue_Cifar100(train=False, transform=transform_test)    
elif args.dataset == 'tinyimagenet':
    print("tinyimagenet")
    trainset = hue_TinyImageNet(transform=transform_train, key=key, ratio=args.ratio, train=True, M=M)    
    testset = hue_TinyImageNet(transform=transform_test, train=False)
elif args.dataset == 'imagenet':
    trainset = hue_ImageNet(transform=transform_train, key=key, ratio=args.ratio, train=True)    
    testset = hue_ImageNet(transform=transform_test, train=False)

torch.save(trainset.getIndices(), presult + "/indices.t7")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

# Model
print('==> Building model..')
print(args.num_classes)
net = torchvision.models.resnet50(num_classes=args.num_classes) if args.dataset == 'tinyimagenet'  else ResNet50(num_classes=args.num_classes)
#net = torchvision.models.vgg19_bn(num_classes=args.num_classes)
#net = torchvision.models.mobilenet_v2(num_classes=args.num_classes)
#net = torchvision.models.shufflenet_v2_x1_0(num_classes=args.num_classes)
#net = torchvision.models.inception_v3(num_classes=args.num_classes)
#net = torchvision.models.densenet161(num_classes=args.num_classes)
#net = torchvision.models.googlenet(num_classes=args.num_classes)
#net = VGG('VGG16')
#net = LeNet()
#net = resnet18()
# net = wide_resnet50_2()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)

log.info(net)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
if args.smooth:
    criterion = LabelSmoothingLoss(args.num_classes, args.smoothing_coef)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(net, d, filename, acc=0):
    if not os.path.isdir(d):
        os.mkdir(d)

    state = {
        'net': net.module.state_dict(),
        'acc': acc
    }
    extension = '.t7'
    torch.save(state, d + '/' + filename + extension )

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if args.adv:
            optimizer.zero_grad()

            x_adv = fgsm_attack(net, inputs, targets, args.epsilon, -1, 1)
            outputs = net(x_adv)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss/len(trainloader), 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    if epoch %10==0:
        save_model(net, presult + '/checkpoint', 'ep_{:02d}_model'.format(epoch), acc)

    if acc > best_acc:
        save_model(net, presult + '/checkpoint', 'best_model', acc)
        best_acc = acc

    return acc

for epoch in range(start_epoch, start_epoch+args.epoch):
    train_loss, train_acc = train(epoch)
    test_acc = test(epoch)

    scheduler.step()
    log.info("Epoch {:4d}, Err {:.4f}, Train Acc {:.4f}, Test Acc {:.4f}, lr {:.4f}, best {:.4f}".format(
        epoch, train_loss, train_acc, test_acc, optimizer.param_groups[0]['lr'], best_acc))
