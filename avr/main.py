
import glob
import random
import argparse
import numpy as np

from dataset import IRavenDataset
from data_utils import ToTensor



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.models as models
from torchvision import transforms, utils

train = IRavenDataset(args.path, "train", args.img_size, transform=transforms.Compose([ToTensor()]),shuffle=True)
valid = IRavenDataset(args.path, "val", args.img_size, transform=transforms.Compose([ToTensor()]))
test = IRavenDataset(args.path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))

subset_indices = np.random.choice(len(train), len(train)*args.perc_train // 100, replace=False)
train_subset = Subset(train, subset_indices)

print("Number of samples in original train set =", len(train))
print("Number of samples in train subset =", len(train_subset))
print("All samples are unique =", len(subset_indices) == len(set(subset_indices)))

trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=16)
validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=16)
testloader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=16)