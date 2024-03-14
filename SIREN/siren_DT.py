import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import os
import cv2
import random
import wandb
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import configargparse
import time
import math

def psnr(ori_img, pred_img):
  max_pixel = 1.0

  mse = ((ori_img-pred_img)**2).mean()

  if(mse == 0):
    return 100

  psnr = 20* math.log10(max_pixel / math.sqrt(mse))

  return psnr

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


def free_image_tensor(sidelength, directory, test_type):
    img = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (sidelength, sidelength), interpolation = cv2.INTER_AREA)
    img = img / 255.0
    img = srgb_to_linear(img)
    img_mean = 0
    img_std = 0
    if test_type == 'gamma0.5':
        img = img ** (1/0.5)
    elif test_type == 'gamma2.0':
        img = img ** (1/2.0)
    elif test_type == 'linear0.5':
        img = img * 0.5
    elif test_type == 'linear2.0':
        img = img * 2.0
    elif test_type == 'standard':
        img_mean = np.mean(img)
        img_std = np.std(img)
        img = (img-img_mean)/img_std
    elif test_type == 'reversed':
        img = 1.0 - img
    elif test_type == 'center1.0':
        img = img - 0.5
    elif test_type == 'center2.0':
        img = (img * 2.0) - 1.0
    elif test_type == 'random_permute':
        img = img.reshape((-1,1))
        encodebook=np.random.permutation(len(img))
        decodebook=np.zeros_like(encodebook)
        for i,val in enumerate(encodebook):
            decodebook[val]=i
        np.save('./decode_book.npy',decodebook)
        img=img[encodebook]
        img=img.reshape((512, 512))

    transform = Compose([
              ToTensor()
            ])
    img = transform(img)
    
    return img, img_mean, img_std

def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)
def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

class ImageFitting(Dataset):
    def __init__(self, sidelength, directory, test_type):
        super().__init__()
        img, img_mean, img_std = free_image_tensor(sidelength, directory, test_type)
        self.length = sidelength * sidelength
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)
        self.mean = img_mean
        self.std =img_std

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.coords[idx], self.pixels[idx]
    
    def give(self, idx):
        return self.coords[idx], self.pixels[idx]

def initialization(seed = 0):   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=int, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=18)
p.add_argument('--lr', type=int, default=-10, help='learning rate. default=2**-10')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--sidelength', type=int, default=512)
p.add_argument('--num_workers', type=int, default=0)
p.add_argument('--project', type=str, default ="Info_18", help = 'Project name')
p.add_argument('--max_steps', type=int, default = 10000)
p.add_argument('--directory', type=str)
p.add_argument('--gpu_num', type=str, default = "0")
p.add_argument('--type', type=str, default = "")

opt = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num
initialization()

step_file = "step_temp/"+str(opt.experiment_name) + opt.type + ".txt"

if os.path.exists(step_file):
    with open(step_file, 'r') as file:
        current = file.read()
        opt.max_steps = int(current)

if(opt.directory == 'DIV2K'):
    directory = opt.directory + "/0" + str(opt.experiment_name) +".png"
else:
    if(opt.experiment_name < 10):
        exper = '0' + str(opt.experiment_name)
    else:
        exper = str(opt.experiment_name)
    directory = opt.directory + "/kodim"+exper+"_"+str(opt.sidelength)+"_grayscale"
    if(opt.directory == "kodak"):
        directory = directory + "_shuffled.png"
    elif(opt.directory == "kodak_zig"):
        directory = directory + "_zigzag.png"
    else:
        directory = directory + ".png"

loaded_data = ImageFitting(opt.sidelength, directory, opt.type)
origin_data = ImageFitting(opt.sidelength, directory, 'test')
full_loader = DataLoader(origin_data, batch_size=2**18, shuffle=False, pin_memory=True, num_workers=opt.num_workers)
shuffle_loader = DataLoader(loaded_data, batch_size=2**18, shuffle=False, pin_memory=True, num_workers=opt.num_workers)

img_siren = Siren(in_features=2, out_features=1, hidden_features=512,
                hidden_layers=3, outermost_linear=True)
img_siren = img_siren.cuda()

# Train
run = wandb.init(
    entity = "opt_bs",
    project=opt.project,
    name = "Kodim"+exper+"_"+str(opt.batch_size)+"_" + str(opt.lr) +"_BIG_55_gray_" + opt.type,
    tags = ["image="+str(opt.project), "batch_size="+str(opt.batch_size), "learning_rate="+str(opt.lr), "gray", opt.type],
    config={
    "learning_rate": opt.lr,
    "dataset": opt.project,
    "batch_size" : 2**opt.batch_size
    }
)

optim = torch.optim.Adam(lr=2**opt.lr, params=img_siren.parameters())

if (opt.type == 'random_permute'):
    model_full, ground_full = next(iter(shuffle_loader))
    model_full, ground_full = model_full.cuda(), ground_full.cuda()
else:
    model_full, ground_full = next(iter(full_loader))
    model_full, ground_full = model_full.cuda(), ground_full.cuda()


step = 0
optim.zero_grad()
end = 1
max_steps = opt.max_steps
batch = 2**opt.batch_size
it = 2**(18-opt.batch_size)

while end:
    idx = 0
    select_checker=np.zeros(model_full.shape[0])
    for idx in range(it):
        
        selected_index=np.random.choice(np.where(select_checker==0)[0],size=batch,replace=False)
        select_checker[selected_index]=1
    
        model_input, ground_truth = loaded_data.give(selected_index)
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()


        model_output, coords = img_siren(model_input)
        
        loss = ((model_output - ground_truth)**2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        img_siren.eval()
        model_test, coords_test = img_siren(model_full)
        if opt.type == 'gamma0.5':
            model_test = torch.clamp(model_test,min=0)
            model_test = model_test ** 0.5
        elif opt.type == 'gamma2.0':
            model_test = torch.clamp(model_test,min=0)
            model_test = model_test ** 2.0
        elif opt.type == 'linear0.5':
            model_test = model_test / 0.5
        elif opt.type == 'linear2.0':
            model_test = model_test / 2.0
        elif opt.type == 'standard':
            model_test = (model_test*loaded_data.std)+loaded_data.mean
        elif opt.type == 'reversed':
            model_test = 1.0 - model_test
        elif opt.type == 'center1.0':
            model_test = model_test + 0.5
        elif opt.type == 'center2.0':
            model_test = (model_test + 1.0) / 2.0
        Test_loss = ((model_test - ground_full)**2).mean()
        Variance = ((model_test - ground_full)**2).var()
        img_siren.train()


        ps = psnr (ground_full, model_test)
        ps_t = psnr(ground_truth, model_output)
        print("step: %d, PSNR: %0.4f, TestLoss: %0.4f, Variance: %0.15f, Train PSNR: %0.4f" % (step, ps, Test_loss, Variance, ps_t))
        wandb.log({"step": step, "PSNR": ps, "loss": loss, "Variance": Variance, "Test_loss": Test_loss, "Train_PSNR": ps_t})
        
        if(ps > 50):                        
            x_type = linear_to_srgb(model_test.cpu().view(opt.sidelength, opt.sidelength).detach().numpy())
            
            if(opt.type == 'random_permute'):
                x_type=x_type.reshape((-1,1))
                decodebook = np.load('./decode_book.npy')
                x_type=x_type[decodebook]
                x_type=x_type.reshape((512, 512))
                
            if(step <= opt.max_steps):
                opt.max_steps = step
            end = 0
            break
        
        if(step >= max_steps):
            end = 0
            break
        step+=1
    
    
wandb.finish()
    
with open(step_file, 'w') as file:
    file.write(str(opt.max_steps))
    