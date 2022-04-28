#!pip install scattering-transform

import os
import glob
import time
import torch
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.models as models
from torchvision import transforms, utils

from PIL import Image
from skimage.transform import resize

import warnings
warnings.filterwarnings("ignore")

from scattering_transform import SCL, SCLTrainingWrapper
from transformers import ViTForImageClassification, ViTFeatureExtractor


class dataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size, transform=None, shuffle=False):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*", "*.npz")) \
                            if dataset_type in f]
        self.img_size = img_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]
        structure = data["structure"]
        meta_target = data["meta_target"]
        meta_structure = data["meta_structure"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = list(range(8))
            np.random.shuffle(indices)
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target
        
        resize_image = []
        for idx in range(0, 16):
            resize_image.append(resize(image[idx,:,:], (self.img_size, self.img_size)))
        resize_image = np.stack(resize_image)

        embedding = torch.zeros((6, 300), dtype=torch.float)
        indicator = torch.zeros(1, dtype=torch.float)
        element_idx = 0
    
        del data
        if self.transform:
            resize_image = self.transform(resize_image)
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target)
            meta_structure = self.transform(meta_structure)
            meta_target = torch.tensor(meta_target, dtype=torch.long)
        return resize_image, target, meta_target, meta_structure, embedding, indicator


class Args:
    
    def __init__(self,):
        self.model = 'ViT_SCL'
        self.epochs = 100
        self.batch_size = 16
        self.seed = 12345
        self.device = 3
        self.load_workers = 16
        self.resume = False
        self.path = '/filer/tmp1/ps851/'
        self.save = './ckpt_res/'
        self.img_size = 80
        self.lr = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.meta_alpha = 0.
        self.meta_beta = 0.
        self.perc_train = 100
        self.verbose = False

        # SCL Hyperparameters
        self.scl_image_size = 224
        self.scl_set_size = 9
        self.scl_conv_channels = [1, 16, 16, 32, 32, 32]
        self.scl_conv_output_dim = 80
        self.scl_attr_heads = 10
        self.scl_attr_net_hidden_dims = [128]
        self.scl_rel_heads = 80
        self.scl_rel_net_hidden_dims = [64, 23, 5]
        
args = Args()

args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)



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



class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.name = args.model
    
    def load_model(self, path, epoch):
        state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch))['state_dict']
        self.load_state_dict(state_dict)

    def save_model(self, path, epoch, acc, loss):
        torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss}, path+'{}_epoch_{}.pth'.format(self.name, epoch))

    def compute_loss(self, output, target, meta_target, meta_structure):
        pass

    def train_(self, image, target, meta_target, meta_structure, embedding, indicator):
        self.optimizer.zero_grad()
        output = self(image, embedding, indicator)
        loss = self.compute_loss(output, target, meta_target, meta_structure)
        loss.backward()
        self.optimizer.step()
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, image, target, meta_target, meta_structure, embedding, indicator):
        with torch.no_grad():
            output = self(image, embedding, indicator)
        loss = self.compute_loss(output, target, meta_target, meta_structure)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def test_(self, image, target, meta_target, meta_structure, embedding, indicator):
        with torch.no_grad():
            output = self(image, embedding, indicator)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return accuracy



from torch._C import device
TO_IMG = transforms.ToPILImage()

def to_image(b):
    b = b.reshape((b.shape[0]*b.shape[1], 1, b.shape[2], b.shape[3]))
    trans_b = [TO_IMG(x) for x in b]
    return trans_b

class ViTSCL(BasicModel):

    def __init__(self, args):
        super(ViTSCL, self).__init__(args)

        self.encoder = ViTFeatureExtractor(do_resize=True).from_pretrained("google/vit-base-patch16-224-in21k")
        self.encoder.image_mean = [0.5]
        self.encoder.image_std = [0.5]

        self.scl = SCL(
            image_size = args.scl_image_size,                           # size of image
            set_size = args.scl_set_size,                               # number of questions + 1 answer
            conv_channels = args.scl_conv_channels,                     # convolutional channel progression, 1 for greyscale, 3 for rgb
            conv_output_dim = args.scl_conv_output_dim,                 # model dimension, the output dimension of the vision net
            attr_heads = args.scl_attr_heads,                           # number of attribute heads
            attr_net_hidden_dims = args.scl_attr_net_hidden_dims,       # attribute scatter transform MLP hidden dimension(s)
            rel_heads = args.scl_rel_heads,                             # number of relationship heads
            rel_net_hidden_dims = args.scl_rel_net_hidden_dims          # MLP for relationship net
        )

        self.decoder = SCLTrainingWrapper(self.scl)

        self.verbose = args.verbose

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x, embedding, indicator):
        questions = x[:, :8, :, :]
        answers = x[:, 8:, :, :]

        if self.verbose:
            print("Shape of questions =", questions.shape)
            print("Shape of answers =", answers.shape)
            print("\n")

        transformed_questions = to_image(questions)
        transformed_answers = to_image(answers)

        if self.verbose:
            print("Length of transformed_questions =", len(transformed_questions))
            print("Length of transformed_answers =", len(transformed_answers))
            # print("Shape of transformed_questions[0] =", transformed_questions[0].shape)
            # print("Shape of transformed_answers[0] =", transformed_answers[0].shape)
            print("\n")

        vit_q = torch.stack([self.encoder(x, return_tensors="pt")['pixel_values'] for x in transformed_questions])
        vit_a = torch.stack([self.encoder(x, return_tensors="pt")['pixel_values'] for x in transformed_answers])

        if self.verbose:
            print("Shape of vit_q =", vit_q.shape)
            print("Shape of vit_a =", vit_a.shape)
            print("\n")

        vit_q = vit_q.reshape((vit_q.shape[0] // 8, 8, vit_q.shape[1], vit_q.shape[2], vit_q.shape[3]))
        vit_a = vit_a.reshape((vit_a.shape[0] // 8, 8, vit_a.shape[1], vit_a.shape[2], vit_a.shape[3]))

        if self.verbose:
            print("Shape of vit_q =", vit_q.shape)
            print("Shape of vit_a =", vit_a.shape)

        vit_q = vit_q.cuda()
        vit_a = vit_a.cuda()

        logits = self.decoder(vit_q, vit_a) # (1, 8) - the logits of each answer being the correct match

        return logits, None


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