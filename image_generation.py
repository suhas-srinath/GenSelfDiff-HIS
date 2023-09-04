import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist
import torch.multiprocessing as mp 
from torchvision import transforms
import utils
from model import DiffusionNet

import os
import shutil
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import numpy as np
from scipy.io import savemat
import cv2

IMG_SIZE = 256
T = 1000
OUT_FOLDER = './gen_images'
os.makedirs(OUT_FOLDER, exist_ok=True)

betas = utils.quadratic_beta_schedule(timesteps=T)
betas_schedule = utils.get_beta_schedule(betas)

def reverse_transforms_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)

@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = utils.get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = utils.get_index_from_list(
        betas_schedule['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = utils.get_index_from_list(betas_schedule['sqrt_recip_alphas'], t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = utils.get_index_from_list(betas_schedule['posterior_variance'], t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, gpu):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=gpu)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=gpu, dtype=torch.long)
        img = sample_timestep(model, img, t)

    return img

def plot(model, gpu):
    all_images = []
    for i in tqdm(range(1000)):
        img = sample_plot_image(model, gpu)
        all_images.append(img)

    for i in range(len(all_images)):
        out_img = reverse_transforms_image(all_images[i].detach().cpu())
        OUT_IMAGE_PATH = os.path.join(OUT_FOLDER, 'image_' + str(i) + '.jpg')
        cv2.imwrite(OUT_IMAGE_PATH, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)) 


gpu = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model = DiffusionNet(dim=64, channels=3).to(gpu)
print("Num params: ", sum(p.numel() for p in model.parameters()))


chkpt_file = '/home/vishnupv/vishnu/MoNuSeg/diffusion/pretrain/snapshots/diff_quadratic/8_0.0001_85_0.03016.pth'
print('Loading checkpoint from:', chkpt_file)
checkpoint = torch.load(chkpt_file)
model.load_state_dict(checkpoint['model_state_dict'])

plot(model, gpu)
