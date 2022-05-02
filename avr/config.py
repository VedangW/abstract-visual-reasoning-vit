class Args:
    
    def __init__(self,):
        self.seed = 12345
        self.cuda = True
        self.distributed = True

        # Model settings
        self.path = '/scratch/vw120/visual_reasoning_data/'
        self.save = './ckpt_res/'
        self.device = 0
        self.load_workers = 16
        self.resume = False
        self.verbose = False

        # Training settings
        self.model = 'ViT_SCL'
        self.epochs = 100
        self.batch_size = 16
        self.perc_train = 100
        self.img_size = 64
        
        # Optimizer settings
        self.lr = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.meta_alpha = 0.
        self.meta_beta = 0.
        
        # ViT parameters
        self.vit_requires_grad = False
        self.vec2image_input_dim = 768

        # # SCL Hyperparameters
        # self.scl_image_size = 224
        # self.scl_set_size = 9
        # self.scl_conv_channels = [1, 16, 16, 32, 32, 32]
        # self.scl_conv_output_dim = 80
        # self.scl_attr_heads = 10
        # self.scl_attr_net_hidden_dims = [128]
        # self.scl_rel_heads = 80
        # self.scl_rel_net_hidden_dims = [64, 23, 5]

        # SCL Hyperparameters
        self.scl_image_size = 224
        self.scl_set_size = 9
        self.scl_conv_channels = [1, 16, 32]
        self.scl_conv_output_dim = 64
        self.scl_attr_heads = 8
        self.scl_attr_net_hidden_dims = [64]
        self.scl_rel_heads = 64
        self.scl_rel_net_hidden_dims = [50, 25, 5]