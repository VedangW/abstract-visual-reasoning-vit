import os
import time
import pickle
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

from torchvision import transforms
from panel_transforms import RotateByAngle, HorizontalFlip, VerticalFlip

from config import Args
from dataset import IRAVENDataset
from models_layers import ViTSCL, BEiTForAbstractVisualReasoning
from utils import batch_to_bin_images

import warnings
warnings.filterwarnings("ignore")


class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

augmentations = [RotateByAngle(90), 
                 RotateByAngle(-90), 
                 RotateByAngle(180), 
                 HorizontalFlip(), 
                 VerticalFlip()]

args = Args()

args.cuda = torch.cuda.is_available()
torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

if not os.path.exists(args.save):
    os.makedirs(args.save)

train = IRAVENDataset(args.path, "train", args.img_size, transform=transforms.Compose([ToTensor()]), shuffle=True)
valid = IRAVENDataset(args.path, "val", args.img_size, transform=transforms.Compose([ToTensor()]))
test = IRAVENDataset(args.path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))

subset_indices = np.random.choice(len(train), len(train)*args.perc_train // 100, replace=False)
train_subset = Subset(train, subset_indices)

print("Number of samples in original train set =", len(train))
print("Number of samples in train subset =", len(train_subset))
print("All samples are unique =", len(subset_indices) == len(set(subset_indices)), "\n")

trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=16)
validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=16)
testloader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=16)

print("Model = %s" % args.model)
model = BEiTForAbstractVisualReasoning(args)
model = model.to(device)
total_trainable_vars = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("{0} trainable variables.".format(total_trainable_vars))


def generate_save_file_path(args):
    fname = args.model +\
            "_perc" + str(args.perc_train) +\
            "_eps" + str(args.epochs) +\
            "_" + time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

    dirname = "experiments/checkpoints/"

    return dirname + fname


save_file = generate_save_file_path(args)
print("Saving results to '" + save_file + "'.\n")

### Helper functions

def train(epoch, save_file, augmentations=None):
    model.train()
    train_loss = 0
    accuracy = 0
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, _, _, _, _) in enumerate(trainloader): 
        counter += 1
        image, target = batch_to_bin_images(image, target, augmentations=augmentations)
        if args.cuda:
            image = image.to(device)
            target = target.to(device)
        loss, acc = model.train_(image, target)
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
    for batch_idx, (image, target, _, _, _, _) in enumerate(validloader):
        counter += 1
        image, target = batch_to_bin_images(image, target)
        if args.cuda:
            image = image.to(device)
            target = target.to(device)
        loss, acc = model.validate_(image, target)
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
    for batch_idx, (image, target, _, _, _, _) in enumerate(testloader):
        counter += 1
        image, target = batch_to_bin_images(image, target)
        if args.cuda:
            image = image.to(device)
            target = target.to(device)
        acc = model.test_(image, target)
        acc_all += acc
    if counter > 0:
        save_str = "Test_: Total Testing Acc: {:.4f}".format((acc_all / float(counter)))
        print(save_str)
        with open(save_file, 'a') as f:
            f.write(save_str + "\n")
    return acc_all/float(counter)


print("Training model...\n")
for epoch in range(0, args.epochs):
    t0 = time.time()
    acc = train(epoch, save_file, augmentations=augmentations)
    avg_loss, avg_acc = validate(epoch, save_file)
    test_acc = test(epoch, save_file)
    model.save_model(args.save, epoch, avg_acc, avg_loss)
    print("Time taken = {:.4f} s\n".format(time.time() - t0))
print("Model trained!\n")