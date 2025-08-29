"""Dataset statistics for normalization.

This module contains pre-computed mean and standard deviation values for various
common datasets, used for data normalization during preprocessing.
"""

CIFAR10 = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
CIFAR100 = dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
ImageNet = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
MNIST = dict(mean=[0.1307], std=[0.3081])
FashionMNIST = dict(mean=[0.2860], std=[0.3530])
STL10 = dict(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713])
SVHN = dict(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
Food101 = dict(mean=[0.5459, 0.4448, 0.3468], std=[0.2761, 0.2691, 0.2821])
Caltech256 = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
OxfordFlowers = dict(mean=[0.434, 0.385, 0.296], std=[0.292, 0.263, 0.272])
OxfordPet = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
SUN397 = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Places365 = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
CelebA = dict(mean=[0.5063, 0.4258, 0.3832], std=[0.2654, 0.2453, 0.2412])
LFW = dict(mean=[0.5063, 0.4258, 0.3832], std=[0.2654, 0.2453, 0.2412])
COCO = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
PascalVOC = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TinyImageNet = dict(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
UCF101 = dict(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
Kinetics400 = dict(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
Cityscapes = dict(
    mean=[0.28689554, 0.32513303, 0.28389177], std=[0.18696375, 0.19017339, 0.18720214]
)
ADE20K = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
NYUDepthV2 = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
CamVid = dict(mean=[0.390687, 0.405213, 0.414304], std=[0.296520, 0.305149, 0.300803])
