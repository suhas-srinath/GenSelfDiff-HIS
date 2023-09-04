import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from scipy.io import loadmat, savemat
from torchsummary import summary
from torch.nn import functional as F
from skimage import io
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.models import resnet18, resnet50
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
import time
import sys
from PIL import Image

from losses import CELoss, Tversky_Loss, FLoss, SSLoss, PolyLogLoss
from diff_seg_model import SegNet


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# seed = 0  # You can choose any value as the seed
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# Hyper Parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_CHANNELS = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 150
NUM_CLASSES = 4
TRAIN_IMAGE_DIR = './train/images'
MODEL_TYPE = 'diff_quadratic_SSFL'
LOSS_TYPE = 'SSFL' + '_' + str(BATCH_SIZE)

transformations = transforms.Compose([
    transforms.ToTensor(),
])


def cleanup():
    dist.destroy_process_group()
    

def my_transforms(image1, mask):
    if random.random() > 0.5:
        image1 = TF.vflip(image1)
        mask = TF.vflip(mask)

    if random.random() > 0.5:
        image1 = TF.hflip(image1)
        mask = TF.hflip(mask)

    if random.random() > 0.7:
        image1 = TF.gaussian_blur(image1, [3, 3], [1.0, 2.0])

    # if random.random() > 0.7:
    #     image1 = TF.adjust_sharpness(image1, 2.0)

    if random.random() > 0.7:
        jitter = transforms.ColorJitter(brightness=.5, contrast=.4)
        image1 = jitter(image1)

    return image1, mask


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


def get_images_list(path1, k=None):
    total_list1 = os.listdir(path1)
    total_list1 = sorted(total_list1, key=lambda x: int(x.split('_')[-1].split('.jpg')[0]))

    if k is None:
        return np.array(total_list1)
    else:
        return np.array(total_list1[:k])


class HN_Dataset(Dataset):
    def __init__(self, image1_dir, image1_list, label_list, transform=None):
        self.image1_dir = image1_dir
        self.image1_list = image1_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        img1_path = os.path.join(self.image1_dir, self.image1_list[index])

        image1 = io.imread(img1_path)
        mask = self.label_list[index]

        if self.transform is not None:
            image1 = self.transform(image1)
            mask = 255 * self.transform(mask)
        image1, mask = my_transforms(image1, mask)

        return image1, mask


def initialize_weights(model):
    # Initializes weights according to the normal distribution
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight.data, 0.0, 0.01)


def F1_score(y_true, y_pred):
    class_f1_scores = []
    _, y_true = torch.max(y_true, 0)
    _, y_pred = torch.max(y_pred, 0)
    for i in range(NUM_CLASSES):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = f1_score(true, pred, zero_division=1)
        class_f1_scores.append(value)

    return class_f1_scores


def image_weights(y_true):
    _, target_label = torch.max(y_true, dim=1)
    img_weights = torch.zeros_like(y_true)
    for i in range(y_true.shape[0]):
        labels, label_counts = torch.unique(target_label[i], return_counts=True)
        label_counts_avg = torch.mean(label_counts / torch.sum(label_counts))
        img_weights[i] = 1.0 + label_counts_avg

        return img_weights


def class_weights(y_predict, y_true):
    fscores = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(y_true.shape[0]):
        fscores += F1_score(y_true[i], y_predict[i])
    # fscores = fscores/y_true.shape[0]
    # weights = (1 - fscores + 0.001) / (fscores + 0.001)

    return fscores


def train_epoch(train_loader, model, optimizer, gpu, epoch):
    model.train()
    losses = []
    p_bar = tqdm(train_loader)

    for h, true_label in p_bar:
        h = h.to(gpu, non_blocking=False)
        true_label = true_label.squeeze(1).to(gpu, non_blocking=False)

        target_label = F.one_hot(true_label.long(), NUM_CLASSES)
        target_label = torch.permute(target_label, (0, 3, 1, 2))
        target_label = target_label.float()

        t = torch.full((h.shape[0],), 0, dtype=torch.long)
        t = t.to(gpu, non_blocking=False)
        predicted_label = model(h, t)

        # ce_loss = CELoss()
        ss_loss = SSLoss()
        f_loss = FLoss(2.0)
        # t_loss = Tversky_Loss(0.75)
        # poly_log_loss = PolyLogLoss(2.0)

        # loss1 = ce_loss(predicted_label, target_label)
        loss2 = ss_loss(predicted_label, target_label)
        loss3 = f_loss(predicted_label, target_label)
        # loss4 = t_loss(predicted_label, target_label)
        # loss5 = poly_log_loss(predicted_label, target_label)
        
        loss = loss2 + loss3

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p_bar.set_description('Epoch {}'.format(epoch))
        p_bar.set_postfix(loss=loss.item())

    print('Epoch: {}\ttotal_loss {:.4f}'.format(epoch, np.mean(losses)))
    return np.mean(losses)


def eval_epoch(eval_loader, model, gpu, epoch, early_stopping=None):
    with torch.no_grad():
        model.eval()
        val_loss = []
        p_bar = tqdm(eval_loader)

        total_items = 0.0
        f_scores = np.array([0.0, 0.0, 0.0, 0.0])
        for h, true_label in p_bar:
            h = h.to(gpu, non_blocking=False)
            true_label = true_label.squeeze(1).to(gpu, non_blocking=False)

            target_label = F.one_hot(true_label.long(), NUM_CLASSES)
            target_label = torch.permute(target_label, (0, 3, 1, 2))
            target_label = target_label.float()

            t = torch.full((h.shape[0],), 0, dtype=torch.long)
            t = t.to(gpu, non_blocking=False)

            predicted_label = model(h, t)
            f_scores += class_weights(predicted_label, target_label)
            total_items += target_label.shape[0]

            # ce_loss = CELoss()
            ss_loss = SSLoss()
            f_loss = FLoss(2.0)
            # t_loss = Tversky_Loss(0.75)
            # poly_log_loss = PolyLogLoss(2.0)

            # loss1 = ce_loss(predicted_label, target_label)
            loss2 = ss_loss(predicted_label, target_label)
            loss3 = f_loss(predicted_label, target_label)
            # loss4 = t_loss(predicted_label, target_label)
            # loss5 = poly_log_loss(predicted_label, target_label)

            loss = loss2 + loss3

            val_loss.append(loss.item())

            p_bar.set_description('Epoch {}'.format(epoch))
            p_bar.set_postfix(loss=loss.item())

    f_scores = f_scores / total_items
    print(f_scores)
    print(np.mean(f_scores[1:]))
    # cls_weights = torch.ones((BATCH_SIZE, NUM_CLASSES, 256, 256)).to(device)
    # for j in range(1, NUM_CLASSES):
    #     cls_weights[:, j, :, :] = 10 * ((1 - f_scores[j] + 0.0001) / (f_scores[j] + 0.0001))

    print('Epoch: {}\tval_loss {:.4f}'.format(epoch, np.mean(val_loss)))
    # early_stopping(np.mean(val_loss), model, epoch)

    return np.mean(val_loss)


def main(gpu, args):
    rank = args['nr'] * args['gpus'] + gpu
    dist.init_process_group('nccl', rank=rank, world_size=args['world_size'])
    torch.cuda.set_device(gpu)

    data_size = None
    backbone = 'Unet'

    # Loading the data
    img1_list = get_images_list(TRAIN_IMAGE_DIR, k=data_size)
    label_file = loadmat('./train/label_patches.mat')
    train_labels = label_file['data']

    ratio = 0.9
    idxs = np.random.RandomState(2023).permutation(img1_list.shape[0])
    split = int(img1_list.shape[0] * ratio)
    train_index = idxs[:split]
    valid_index = idxs[split:]

    train_dataset = HN_Dataset(TRAIN_IMAGE_DIR,
                                  img1_list[train_index], train_labels[train_index],
                                  transform=transformations)
    eval_dataset = HN_Dataset(TRAIN_IMAGE_DIR,
                                 img1_list[valid_index], train_labels[valid_index],
                                 transform=transformations)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args['world_size'],
                                                                    rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  drop_last=True, num_workers=4, pin_memory=True,
                                  sampler=train_sampler)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,
                                                                   num_replicas=args['world_size'],
                                                                   rank=rank, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, drop_last=False, num_workers=4, pin_memory=True,
                                 sampler=eval_sampler)

    model = SegNet(dim=64, channels=3, num_classes=4).to(gpu)
    initialize_weights(model)

    start_epoch = 0
    if args['load_from_chkpt'] is not None:
        chkpt_file = args['load_from_chkpt']
        print('Loading checkpoint from:', chkpt_file)
        checkpoint = torch.load(chkpt_file, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint['model_state_dict']
        new_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k[:15] != 'net.final_conv.'}
        model.load_state_dict(new_pretrained_dict, strict=False)
        # model.load_state_dict(pretrained_dict)

    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    checkpoint_path = args['checkpoints_path']
    os.makedirs(checkpoint_path, exist_ok=True)

    if gpu == 0:
        early_stopping = EarlyStopping(patience=15, verbose=True,
                                       path=checkpoint_path + '{}_{}.pth'.format(BATCH_SIZE, LEARNING_RATE))
    else:
        early_stopping = None

    train_losses = []
    eval_losses = []
    start_time = time.process_time()
    for epoch in range(start_epoch, EPOCHS):
        print('epoch {}/{}'.format(epoch + 1, EPOCHS))
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(train_dataloader, model, optimizer, gpu, epoch + 1)
        eval_loss = eval_epoch(eval_dataloader, model, gpu, epoch + 1, early_stopping)

        mean_eval_loss = torch.tensor(eval_loss / args['gpus']).to(gpu)
        mean_train_loss = torch.tensor(train_loss / args['gpus']).to(gpu)
        dist.barrier()
        dist.all_reduce(mean_eval_loss)
        dist.all_reduce(mean_train_loss)
        print('gpu {} eval_loss:{}, mean_loss:{}'.format(gpu, eval_loss,
                                                         mean_eval_loss.cpu().numpy()))

        # if optim_name.split('-')[-1] == 'step':
        #     scheduler.step(mean_eval_loss.cpu().numpy())
        # elif optim_name.split('-')[-1] == 'cosine':
        #     scheduler.step()
        # elif optim_name.split('-')[-1] == 'no':
        #     pass

        if gpu == 0:
            early_stopping(mean_eval_loss.cpu().numpy(), model, epoch + 1)
        '''
        if early_stopping.early_stop:
            print('Early stop!')
            break
        '''
        train_losses.append(mean_train_loss.cpu().numpy())
        eval_losses.append(mean_eval_loss.cpu().numpy())
    
    current_time = time.process_time()
    print("Total Time Elapsed={:12.5} seconds".format(str(current_time - start_time)))
    
    # saving the plots
    plots_path = './plots/' + MODEL_TYPE
    os.makedirs(plots_path, exist_ok=True)
    epochs = np.arange(start_epoch, EPOCHS)
    train_losses = np.array(train_losses)
    eval_losses = np.array(eval_losses)
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    axes.plot(epochs, train_losses, 'tab:blue', epochs, eval_losses, 'tab:orange')
    axes.set_title(f'Training and Validation Loss (pretrained model = diffusion, loss = (SS + Focal) Loss, '
                   f'data size = {data_size})',
                   weight='bold', fontsize=7)
    axes.set_xlabel('Epochs', weight='bold', fontsize=9)
    axes.set_ylabel('Loss', weight='bold', fontsize=9)
    axes.legend(['training loss', 'validation loss'], loc='best')
    plt.savefig(plots_path + '/' + backbone + '_' + LOSS_TYPE + 'loss_' + str(data_size) + '.jpg', dpi=300)

    cleanup()


if __name__ == '__main__':
    args = {}

    args['gpus'] = 1
    args['nr'] = 0
    args['world_size'] = args['gpus']
    args['checkpoints_path'] = './snapshots/' + MODEL_TYPE + '/'
    args['load_from_chkpt'] = '../pretrain/snapshots/diff_quadratic_general/6_0.0001_100_0.02042.pth'

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    print(args['gpus'])
    mp.spawn(main, args=(args,), nprocs=args['gpus'])
