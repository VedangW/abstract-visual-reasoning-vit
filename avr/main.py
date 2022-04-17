import time
import argparse
import numpy as np

from data_utils import ToTensor
from dataset import IRavenDataset

from training import TrainerAndEvaluator
from models import CnnLstm

import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from torchvision import transforms


parser = argparse.ArgumentParser(description='our_model')

parser.add_argument('--model', type=str, default='CNN_LSTM')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--load_workers', type=int, default=16)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--path', type=str, default='./data/IRAVEN/')
parser.add_argument('--save', type=str, default='./experiments/checkpoint/')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--meta_alpha', type=float, default=0.)
parser.add_argument('--meta_beta', type=float, default=0.)
parser.add_argument('--perc_train', type=float, default=100)

args = parser.parse_args()


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

model = CnnLstm(args)
model = model.cuda()

SAVE_FILE = "fin3_lstm_dat100_eps200_" + time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime()) + "_" + str(args.perc_train)

trainer = TrainerAndEvaluator(model)

for epoch in range(0, args.epochs):
    t0 = time.time()
    trainer.train(trainloader, epoch, args, SAVE_FILE)
    avg_loss, avg_acc = trainer.validate(validloader, args, SAVE_FILE)
    test_acc = trainer.test(testloader, args, SAVE_FILE)
    trainer.model.save_model(args.save, epoch, avg_acc, avg_loss)
    print("Time taken = {:.4f} s\n".format(time.time() - t0))