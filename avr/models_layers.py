import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from transformers import BeitConfig, BeitModel, ViTForImageClassification
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

    def compute_loss(self, output, target):
        pass

    def train_(self, image, target):
        self.optimizer.zero_grad()
        output = self(image)
        loss = self.compute_loss(output, target)
        loss.backward()
        self.optimizer.step()
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, image, target):
        with torch.no_grad():
            output = self(image)
        loss = self.compute_loss(output, target)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def test_(self, image, target):
        with torch.no_grad():
            output = self(image)
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

    def compute_loss(self, output, target):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x):
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


class BEiTForAbstractVisualReasoning(BasicModel):

    def __init__(self, args, num_classes=8, fc_layer_sizes=[64, 32, 8], beit_patch_size=80, 
                 beit_num_channels=1, beit_image_size=240, beit_freeze=True, 
                 beit_freeze_perc=60, beit_pretrained_ckpt='microsoft/beit-base-patch16-224', 
                 verbose=True) -> None:
        super(BEiTForAbstractVisualReasoning, self).__init__(args)

        self.verbose = verbose
        self.num_classes = num_classes

        # BEiT

        self.beit_freeze = beit_freeze
        self.beit_freeze_perc = beit_freeze_perc
        self.beit_patch_size = beit_patch_size
        self.beit_num_channels = beit_num_channels
        self.beit_image_size = beit_image_size
        self.beit_pretrained_ckpt = beit_pretrained_ckpt

        self.beit_config = BeitConfig(patch_size=self.beit_patch_size, 
                                      num_channels=self.beit_num_channels, 
                                      image_size=self.beit_image_size).\
                                      from_pretrained(self.beit_pretrained_ckpt)

        self.beit = BeitModel(self.beit_config)

        if self.beit_freeze:
            print("Freezing first {0}% of parameters of BEiT.".format(self.beit_freeze_perc))

            self.beit_total_named_params = len(list(self.beit.named_parameters()))
            self.unfreeze_last = self.beit_total_named_params - \
                (self.beit_total_named_params*self.beit_freeze_perc//100)
            
            print("Last {0} parameters remain unfrozen.".format(self.unfreeze_last))

            for name, param in list(self.beit.named_parameters())[:-self.unfreeze_last]:
                param.requires_grad = False

        # MLP

        self.fc_layer_sizes = fc_layer_sizes

        if not self.fc_layer_sizes:
            raise ValueError("Need at least 1 FC layer!")

        if self.fc_layer_sizes[0] != 768:
            self.fc_layer_sizes = [768] + self.fc_layer_sizes

        if self.fc_layer_sizes[-1] != 1 and self.fc_layer_sizes[-1] > 1:
            self.fc_layer_sizes += [1]
        elif self.fc_layer_sizes[-1] < 1:
            raise ValueError("Invalid fc_layer_sizes!")

        self.mlp_layers = nn.ModuleList([])

        for i in range(len(self.fc_layer_sizes)-1):
            self.mlp_layers.append(nn.Linear(self.fc_layer_sizes[i], self.fc_layer_sizes[i+1]))
            if i != len(self.fc_layer_sizes)-2:
                self.mlp_layers.append(nn.ReLU())

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)


    def compute_loss(self, output, target, test_mode=False):
        if not test_mode:
            pred = output[0].squeeze().float()
        else:
            pred = torch.stack(output).reshape(-1)
            target = F.one_hot(target.to(torch.int64), 
                               num_classes=self.num_classes).reshape(-1)

        loss = self.criterion(pred, target.float())
        return loss

    def forward(self, x, output_attentions=False):
        beit_outs = self.beit(x, output_attentions=output_attentions)

        x_mlp = beit_outs['last_hidden_state'][:, 0, :]

        if output_attentions and 'attentions' in beit_outs.keys():
            attns = beit_outs['attentions']
        else:
            attns = None

        for layer in self.mlp_layers:
            x_mlp = layer(x_mlp)

        return x_mlp, attns

    def predict_ranking(self, outputs):
        return torch.stack([torch.argmax(x.squeeze()) for x in outputs])

    def train_(self, images, target):
        self.optimizer.zero_grad()
        output = self(images)
        loss = self.compute_loss(output, target)
        loss.backward()
        self.optimizer.step()
        pred = torch.round(torch.sigmoid(output[0].squeeze()))
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, images, target):
        target = torch.stack(target) if isinstance(target, list) else target 
        with torch.no_grad():
            outputs = [self(image)[0] for image in images]
        loss = self.compute_loss(outputs, target, test_mode=True)
        pred = self.predict_ranking(outputs=outputs)
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def test_(self, images, target):
        target = torch.stack(target) if isinstance(target, list) else target
        with torch.no_grad():
            outputs = [self(image)[0] for image in images]
        pred = self.predict_ranking(outputs=outputs)
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return accuracy
