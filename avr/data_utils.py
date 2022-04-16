import os
import torch

def gen():
    file_list = []
    for root, dir, files in os.walk("/common/home/kp956/CS536/I-RAVEN/test/"):
        for file in files:
            if file.endswith(".npz"):
                file_list.append(os.path.join(root,file))
    return file_list

class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)