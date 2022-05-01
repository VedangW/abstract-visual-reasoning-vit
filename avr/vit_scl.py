import os
import time
import torch
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from torchvision import transforms

from config import Args
from dataset import dataset
from models_layers import ViTSCL

import warnings
warnings.filterwarnings("ignore")

### Helper functions

def train(epoch, save_file):
    model.train()
    train_loss = 0
    accuracy = 0
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(trainloader):
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()
        loss, acc = model.train_(image, target, meta_target, meta_structure, embedding, indicator)
        save_str = 'Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}'.format(epoch, batch_idx, loss, acc)
        if counter % 20 == 0:
            print(save_str)
        with open(save_file, 'a') as f:
            f.write(save_str + "\n")
        loss_all += loss
        acc_all += acc
    if counter > 0:
        save_str = "Train_: Avg Training Loss: {:.6f}, Avg Training Acc: {:.6f}".format(
            loss_all/float(counter),
            (acc_all/float(counter))
        )
        print(save_str)
        with open(save_file, 'a') as f:
            f.write(save_str + "\n")
    return loss_all/float(counter), acc_all/float(counter)

def validate(epoch, save_file):
    model.eval()
    val_loss = 0
    accuracy = 0
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    batch_idx = 0
    for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(validloader):
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()
        loss, acc = model.validate_(image, target, meta_target, meta_structure, embedding, indicator)
        loss_all += loss
        acc_all += acc
    if counter > 0:
        save_str = "Val_: Total Validation Loss: {:.6f}, Acc: {:.4f}".format((loss_all/float(counter)), (acc_all/float(counter)))
        print(save_str)
        with open(save_file, 'a') as f:
            f.write(save_str + "\n")
    return loss_all/float(counter), acc_all/float(counter)

def test(epoch, save_file):
    model.eval()
    accuracy = 0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(testloader):
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()
        acc = model.test_(image, target, meta_target, meta_structure, embedding, indicator)
        acc_all += acc
    if counter > 0:
        save_str = "Test_: Total Testing Acc: {:.4f}".format((acc_all / float(counter)))
        print(save_str)
        with open(save_file, 'a') as f:
            f.write(save_str + "\n")
    return acc_all/float(counter)

class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)


args = Args()

args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

train = dataset(args.path, "train", args.img_size, transform=transforms.Compose([ToTensor()]),shuffle=True)
valid = dataset(args.path, "val", args.img_size, transform=transforms.Compose([ToTensor()]))
test = dataset(args.path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))

subset_indices = np.random.choice(len(train), len(train)*args.perc_train // 100, replace=False)
train_subset = Subset(train, subset_indices)

print("Number of samples in original train set =", len(train))
print("Number of samples in train subset =", len(train_subset))
print("All samples are unique =", len(subset_indices) == len(set(subset_indices)))

trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=16)
validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=16)
testloader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=16)

model = ViTSCL(args)
model = model.cuda()

SAVE_FILE = "ViTSCL_dat100_eps200_" + time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime()) + "_" + str(args.perc_train)

for epoch in range(0, args.epochs):
    t0 = time.time()
    train(epoch, SAVE_FILE)
    avg_loss, avg_acc = validate(epoch, SAVE_FILE)
    test_acc = test(epoch, SAVE_FILE)
    model.save_model(args.save, epoch, avg_acc, avg_loss)
    print("Time taken = {:.4f} s\n".format(time.time() - t0))