import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from skimage import io
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random

IMG_SIZE = 256


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def quadraple_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.25, beta_end ** 0.25, timesteps) ** 4


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, betas_schedule, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(betas_schedule['sqrt_alphas_cumprod'], t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        betas_schedule['sqrt_one_minus_alphas_cumprod'], t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def _my_normalization(x):
    return (x * 2) - 1


def get_images_list(path1, k=None):
    total_list1 = os.listdir(path1)
    total_list1 = sorted(total_list1, key=lambda x: int(x.split('_')[-1].split('.jpg')[0]))
    if k is None:
        return np.array(total_list1)
    else:
        return np.array(total_list1[:k])


class GlaS_Dataset(Dataset):
    def __init__(self, image1_dir, image1_list, transform=None):
        self.image1_dir = image1_dir
        self.image1_list = image1_list
        self.transform = transform

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        img1_path = os.path.join(self.image1_dir, self.image1_list[index])

        image1 = io.imread(img1_path)

        if self.transform is not None:
            image1 = self.transform(image1)

        return image1


def load_transformed_dataset():
    data_transforms = [
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(_my_normalization)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    data_size = None
    TRAIN_IMAGE_DIR = '../unlabelled_img_patches'
    img1_list = get_images_list(TRAIN_IMAGE_DIR, k=data_size)

    ratio = 0.9
    idxs = np.random.RandomState(2023).permutation(img1_list.shape[0])
    split = int(img1_list.shape[0] * ratio)
    train_index = idxs[:split]
    valid_index = idxs[split:]

    train_dataset = GlaS_Dataset(TRAIN_IMAGE_DIR, img1_list[train_index], transform=data_transform)
    eval_dataset = GlaS_Dataset(TRAIN_IMAGE_DIR, img1_list[valid_index], transform=data_transform)

    return train_dataset, eval_dataset


def reverse_transforms_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


def get_beta_schedule(betas):
    schedule = {}
    schedule['alphas'] = 1. - betas
    schedule['alphas_cumprod'] = torch.cumprod(schedule['alphas'], dim=0)
    schedule['alphas_cumprod_prev'] = F.pad(schedule['alphas_cumprod'][:-1], (1, 0), value=1.0)
    schedule['sqrt_recip_alphas'] = torch.sqrt(1.0 / schedule['alphas'])
    schedule['sqrt_alphas_cumprod'] = torch.sqrt(schedule['alphas_cumprod'])
    schedule['sqrt_one_minus_alphas_cumprod'] = torch.sqrt(1. - schedule['alphas_cumprod'])
    schedule['posterior_variance'] = betas * (1. - schedule['alphas_cumprod_prev']) / (
            1. - schedule['alphas_cumprod'])
    return schedule


def get_loss(noise, noise_pred, time_stamps, betas_schedule, gpu):
    t = time_stamps.cpu()
    snr = 1.0 / (1 - betas_schedule['alphas_cumprod'][t]) - 1
    k = 1.0
    gamma = 1.0
    lambda_t = 1.0 / ((k + snr) ** gamma)
    lambda_t = lambda_t.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(gpu)

    n = noise.shape[1] * noise.shape[2] * noise.shape[3]
    loss = torch.sum(lambda_t * F.mse_loss(noise, noise_pred, reduction='none')) / n
    return loss


# def get_loss(noise, noise_pred, time_stamps, betas_schedule, gpu):
#     t = time_stamps.cpu()
#     n = noise.shape[1] * noise.shape[2] * noise.shape[3]

#     snr = 1.0 / (1 - betas_schedule['alphas_cumprod'][t]) - 1
#     k = 1.0
#     gamma = 1.0
#     lambda_t = 1.0/((k+snr)**gamma)
#     lambda_t = lambda_t.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(gpu)
#     loss1 = torch.sum(lambda_t * F.mse_loss(noise, noise_pred, reduction='none'))/n

#     loss2 = torch.sum(F.mse_loss(noise, noise_pred, reduction='none'))/n

#     c = 0.001
#     loss = loss1 + c*loss2
#     return loss


# def get_loss(noise, noise_pred, time_stamps, betas_schedule, gpu):
#     t = time_stamps.cpu()
#     n = noise.shape[1] * noise.shape[2] * noise.shape[3]

#     snr = 1.0 / (1 - betas_schedule['alphas_cumprod'][t]) - 1
#     k = 1.0
#     gamma = 1.0
#     lambda_t = 1.0/((k+snr)**gamma)
#     lambda_t = lambda_t.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(gpu)
#     loss1 = torch.sum(lambda_t * F.mse_loss(noise, noise_pred, reduction='none'))/n

#     scale_factor = (1.0 - betas_schedule['alphas'][t]) / (betas_schedule['alphas'][t] * (1.0 - betas_schedule['alphas_cumprod'][t]))
#     scale_factor = scale_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(gpu)
#     loss2 = torch.sum(scale_factor * F.mse_loss(noise, noise_pred, reduction='none'))/n

#     c = 0.001
#     loss = loss1 + c*loss2
#     return loss


# def get_loss(noise, noise_pred, time_stamps, betas_schedule, gpu):
#     t = time_stamps.cpu()
#     n = noise.shape[1] * noise.shape[2] * noise.shape[3]

#     loss = torch.sum(F.mse_loss(noise, noise_pred, reduction='none'))/n

#     return loss


# def get_loss(noise, noise_pred, time_stamps, betas_schedule, gpu):
#     t = time_stamps.cpu()
#     n = noise.shape[1] * noise.shape[2] * noise.shape[3]

#     snr = 1.0 / (1 - betas_schedule['alphas_cumprod'][t]) - 1
#     k = 1.0
#     gamma = 1.0
#     lambda_t = 1.0/((k+snr)**gamma)
#     lambda_t = lambda_t.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(gpu)
#     loss1 = torch.sum(lambda_t * F.mse_loss(noise, noise_pred, reduction='none'))/n

#     scale_factor = (1.0 - betas_schedule['alphas'][t] + 0.0001) / (2 * betas_schedule['alphas'][t] * (1.0 - betas_schedule['alphas_cumprod_prev'][t]) + 0.0001)
#     scale_factor = scale_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(gpu)
#     loss2 = torch.sum(scale_factor * F.mse_loss(noise, noise_pred, reduction='none'))/n

#     loss = loss1 + loss2


#     return loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0,
                 path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch=None, ddp=False):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, ddp)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, ddp)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, ddp):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if epoch != None:
            weight_path = self.path[:-4] + '_' + str(epoch) + '_' + str(val_loss)[:7] + '.pth'
        else:
            weight_path = self.path

        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'model_state_dict': model.module.state_dict(),
        }, weight_path)

        self.val_loss_min = val_loss


