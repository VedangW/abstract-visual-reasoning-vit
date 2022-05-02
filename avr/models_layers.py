import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from transformers import ViTForImageClassification
from scattering_transform import SCL, SCLTrainingWrapper


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


def zeros(shape):
    return nn.init.zeros_(torch.empty(shape))

def glorot(shape):
    return nn.init.xavier_uniform_(torch.empty(shape), gain=1.)

class Vec2Image(nn.Module):

    def __init__(self, input_dim, output_dim, bias=True, act=F.relu):
        super(Vec2Image, self).__init__()

        if len(output_dim) != 3:
            raise ValueError("output_dim must be 3d.")

        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act

        self.weight = nn.Parameter(glorot((input_dim, output_dim[1]*output_dim[2])))

        if bias:
            self.bias = nn.Parameter(zeros((output_dim[1]*output_dim[2])))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Expected shape of x: (b, n)

        x = self.act(torch.matmul(x, self.weight) + self.bias)
        x = x.view((x.shape[0] // 8, 8, 1, self.output_dim[1], self.output_dim[2]))

        return x


TO_IMG = transforms.ToPILImage()

def to_image(b):
    b = b.reshape((b.shape[0]*b.shape[1], 1, b.shape[2], b.shape[3]))
    trans_b = [TO_IMG(x) for x in b]
    return trans_b

class ViTSCL(BasicModel):

    def __init__(self, args):
        super(ViTSCL, self).__init__(args)

        self.id2label = {'opt' + str(k): k for k in range(8)}
        self.label2id = {k: 'opt' + str(k) for k in range(8)}

        self.encoder = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                                num_labels=8,
                                                                id2label=self.id2label,
                                                                label2id=self.label2id,
                                                                output_hidden_states=True) 

        for _, param in self.encoder.named_parameters():
            if param.requires_grad:
                param.requires_grad = args.vit_requires_grad

        self.vec2image = Vec2Image(input_dim=args.vec2image_input_dim, output_dim=(1, args.scl_image_size, args.scl_image_size))

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

        # Transforms
        self.grayscale_to_rgb = transforms.Lambda(lambda x: x.reshape((1, x.shape[0], x.shape[1])).repeat(3, 1, 1))
        self.reshape_input_batch = transforms.Lambda(lambda x: x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3])))
        self.resize_to_vit_size = transforms.Resize((224, 224))

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x, embedding, indicator):
        questions = x[:, :8, :,  :]
        answers = x[:, 8:, :, :]

        q = self.reshape_input_batch(questions)
        a = self.reshape_input_batch(answers)

        q = torch.stack([self.resize_to_vit_size(self.grayscale_to_rgb(x)) for x in q])
        a = torch.stack([self.resize_to_vit_size(self.grayscale_to_rgb(x)) for x in a])

        q_vit = self.encoder(q)['hidden_states'][-1][:, 0, :]
        a_vit = self.encoder(a)['hidden_states'][-1][:, 0, :]

        q_imgs = self.vec2image(q_vit)
        a_imgs = self.vec2image(a_vit)

        logits = self.decoder(q_imgs, a_imgs)

        return logits, None