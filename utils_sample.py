# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:30:16 2023

@author: Admin
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from tqdm import tqdm
import os
import random
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms as T

WEIGHT_CLIP = 0.01

#%%
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

    def __call__(self, val_loss, model1=None, model2=None, model3=None, model4=None, model5=None, epoch=None, ddp=False):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model1, model2, model3, model4, model5, epoch, ddp)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if(epoch%3 == 0):
                self.save_checkpoint(val_loss, model1, model2, model3, model4, model5, epoch, ddp)

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model1, model2, model3, model4, model5, epoch, ddp)
            self.counter = 0

    def save_checkpoint(self, val_loss, model1, model2, model3, model4, model5, epoch, ddp):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if epoch != None:
            weight_path1 = self.path[:-4] + '_' + 'Ek_' + str(epoch) + '_' + str(val_loss)[:7] + '.pth'
            weight_path2 = self.path[:-4] + '_' + 'Ea_' + str(epoch) + '_' + str(val_loss)[:7] + '.pth'
            weight_path3 = self.path[:-4] + '_' + 'G_' + str(epoch) + '_' + str(val_loss)[:7] + '.pth'
            weight_path4 = self.path[:-4] + '_' + 'D_' + str(epoch) + '_' + str(val_loss)[:7] + '.pth'
            weight_path5 = self.path[:-4] + '_' + 'Phi_' + str(epoch) + '_' + str(val_loss)[:7] + '.pth'
        else:
            weight_path1 = self.path[:-4] + 'Ek_' + '.pth'
            weight_path2 = self.path[:-4] + 'Ea_' + '.pth'
            weight_path3 = self.path[:-4] + 'G_' + '.pth'
            weight_path4 = self.path[:-4] + 'D_' + '.pth'
            weight_path5 = self.path[:-4] + 'Phi_' + '.pth'
            
        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'model_state_dict': model1.state_dict(),
        }, weight_path1)
        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'model_state_dict': model2.state_dict(),
        }, weight_path2)
        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'model_state_dict': model3.state_dict(),
        }, weight_path3)
        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'model_state_dict': model4.state_dict(),
        }, weight_path4)
        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'model_state_dict': model5.state_dict(),
        }, weight_path5)
        
        self.val_loss_min = val_loss   


#%%
class LPW_Dataset(Dataset):
    def __init__(self, train_subject_frame_eye, train_images, landmark_priors, landmark_priors_positions, transform=None):
        self.train_subject_frame_eye = train_subject_frame_eye
        self.train_images = train_images
        self.landmark_priors = landmark_priors
        self.landmark_priors_positions = landmark_priors_positions
        self.transform = transform
        self.landmark_prior_size = len(self.landmark_priors)
        
    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):

        I1, lm_prior, lm_prior_position = self.__select_sample(index)
        
        if self.transform is not None:
            I1 = self.transform(I1)
            lm_prior = self.transform(lm_prior)
            lm_prior_position = T.ToTensor()(lm_prior_position)

        return I1, lm_prior, lm_prior_position   
    
    def __select_sample(self, index):
        
        landmark_prior_index = random.randint(0, self.landmark_prior_size - 1)
    
        I1 = self.train_images[index]
    
        return I1, self.landmark_priors[landmark_prior_index], self.landmark_priors_positions[landmark_prior_index]


#%%    
def train_epoch(train_loader, Phi, VGG_model, Ek, Ea, G, D, Phi_optimizer, Ek_Ea_G_optimizer, D_optimizer, gpu, epoch, adv_weight):
    Ek.train()
    Ea.train()
    G.train()
    D.train()
    Phi.train()
    
    Ek_Ea_G_losses = []
    D_losses = []
    Phi_losses = []
    p_bar = tqdm(train_loader)
    
    
    running_idx = 1
    criterion = nn.BCELoss()
    
    width, height = 100, 70
    x, y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
    points = torch.cat((y.reshape(-1, 1), x.reshape(-1, 1)), dim=1)
    points = points.to(gpu)
    
    
    for I1, K, V_real in p_bar:
        I1 = I1.to(gpu) # Input Image
        K = K.to(gpu)   # Landmark Prior Representation
        V_real = V_real.float().to(gpu) # Landmark Prior Position
        V_real = torch.reshape(V_real, (V_real.shape[0], -1))
        
        Ek_I1 = Ek(I1)
        V_predict = Phi(Ek_I1)
        V_predict = torch.reshape(V_predict, (V_predict.shape[0], 2, -1))
        
        positions_to_keypoint_representation = Positions_To_KeypointRepresentation()
        K_hat = positions_to_keypoint_representation(V_predict, points)
        K_hat = K_hat.unsqueeze(1)

        I1_hat = G(K_hat, Ea(I1))
        
        Recon_loss = ReconstructLoss(VGG_model)
        mse_loss = nn.MSELoss()
        L1_loss = nn.L1Loss()
        Adv_loss = AdversarialLoss(D)
        
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # disc_real = D(K).reshape(-1)
        # loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        # disc_Ek_I1 = D(Ek_I1.detach()).reshape(-1)
        # loss_disc_Ek_I1 = criterion(disc_Ek_I1, torch.zeros_like(disc_Ek_I1))
        # D_loss = (loss_disc_real + loss_disc_Ek_I1)
        D_loss = -1*Adv_loss(Ek_I1.detach(), K)
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        
        # clip critic weights between -0.01, 0.01
        for p in D.parameters():
           p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
              
        # Training the landmarks encoder Ek, appearance encoder Ea, and the generator G
        rec_loss = Recon_loss(I1, I1_hat)
        cons_loss = L1_loss(Ek_I1, K_hat) # cons_loss = L1_loss(Ek_I1.reshape(K_hat.shape[0], -1), K_hat)
        # disc_Ek_I1 = D(Ek_I1).reshape(-1)
        # adv_loss = criterion(disc_Ek_I1, torch.ones_like(disc_Ek_I1))
        adv_loss = -1*torch.mean(D(Ek_I1))
        
        Ek_Ea_G_loss = rec_loss + (2*cons_loss) + (adv_weight*adv_loss)

        Ek_Ea_G_optimizer.zero_grad()
        Ek_Ea_G_loss.backward(retain_graph=True)
        Ek_Ea_G_optimizer.step()
        
        # Training the regressor using consistent loss, regressor loss
        consistent_loss = L1_loss(K_hat, Ek_I1.detach())
        # consistent_loss = L1_loss(K_hat, Ek_I1.detach().reshape(K_hat.shape[0], -1))
        regressor_loss = mse_loss(Phi(K), V_real)
        Phi_loss = consistent_loss + (0.5*regressor_loss)
        
        Phi_optimizer.zero_grad()
        Phi_loss.backward()
        Phi_optimizer.step()
         
        
        if running_idx % 500 == 0:

            fig, axs = plt.subplots(2, 3, figsize=(24, 12))

            I1_set = 255 * torch.permute(I1, (0, 2, 3, 1))            
            I1_img = np.uint8(I1_set[0].detach().cpu().numpy())
            axs[0, 0].imshow(I1_img, cmap='gray')
            axs[0, 0].set_title(r'$I_{1}$')

            K_set = 255 * torch.permute(Ek_I1, (0, 2, 3, 1))            
            K_img = np.uint8(K_set[0].detach().cpu().numpy())
            axs[0, 1].imshow(K_img, cmap='gray')
            axs[0, 1].set_title(r'$E_k(I_{1})$')

            K_hat_set = 255 * torch.permute(K_hat, (0, 2, 3, 1))    
            # K_hat_set = 255 * torch.permute(K_hat.reshape(K_hat.shape[0], 1, height, width), (0, 2, 3, 1)) 
            K_hat_img = np.uint8(K_hat_set[0].detach().cpu().numpy())
            axs[0, 2].imshow(K_hat_img, cmap='gray')
            axs[0, 2].set_title(r'$\hat{K}$')

            Ki_set = 255 * torch.permute(K, (0, 2, 3, 1))            
            Ki_img = np.uint8(Ki_set[0].detach().cpu().numpy())
            axs[1, 0].imshow(Ki_img, cmap='gray')
            axs[1, 0].set_title(r'$K_{real}$')

            Ea_set = 255 * torch.permute(Ea(I1), (0, 2, 3, 1))    
            Ea_img = np.uint8(Ea_set[0].detach().cpu().numpy())
            axs[1, 1].imshow(Ea_img, cmap='gray')
            axs[1, 1].set_title(r'$E_a(I_{1})$')

            I1_hat_set = 255 * torch.permute(I1_hat, (0, 2, 3, 1))            
            I1_hat_img = np.uint8(I1_hat_set[0].detach().cpu().numpy())
            axs[1, 2].imshow(I1_hat_img, cmap='gray')
            axs[1, 2].set_title(r'$\hat{I}_{1}$')

            fig.suptitle(f'Training <-----> Epoch={epoch} and Iter={running_idx}')
            
            plt.show()
                        
            
        running_idx += 1 
        
        Ek_Ea_G_losses.append(Ek_Ea_G_loss.item())
        D_losses.append(D_loss.item())
        Phi_losses.append(Phi_loss.item())

        p_bar.set_description('Epoch {}'.format(epoch))
        p_bar.set_postfix(rec_loss=rec_loss.item(), D_loss=D_loss.item(), adv_loss=adv_loss.item(), 
                          reg_loss=regressor_loss.item(), consistent_loss=consistent_loss.item())

    print('Epoch: {} \t total_Ek_Eg_G_loss {:.4f}, \t total_D_loss {:.4f}, \t total_Phi_loss {:.4f}'.format(epoch, 
                                                                                                          np.mean(Ek_Ea_G_losses), 
                                                                                                          np.mean(D_losses), 
                                                                                                          np.mean(Phi_losses)))
    return np.mean(Ek_Ea_G_losses), np.mean(D_losses), np.mean(Phi_losses)


#%%
def eval_epoch(eval_loader, Phi, VGG_model, Ek, Ea, G, D, gpu, epoch, adv_weight, early_stopping=None):
    with torch.no_grad():
        Ek.eval()
        Ea.eval()
        G.eval()
        D.eval()
        Phi.eval()
        
        Ek_Ea_G_losses = []
        D_losses = []
        Phi_losses = []
        p_bar = tqdm(eval_loader)
        
        running_idx = 1
        criterion = nn.BCELoss()
        
        width, height = 100, 70
        x, y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        points = torch.cat((y.reshape(-1, 1), x.reshape(-1, 1)), dim=1)
        points = points.to(gpu)

        for I1, K, V_real in p_bar:
            I1 = I1.to(gpu) # Input Image
            K = K.to(gpu)   # Landmark Prior Representation
            V_real = V_real.float().to(gpu) # Landmark Prior Position
            V_real = torch.reshape(V_real, (V_real.shape[0], -1))
            
            Ek_I1 = Ek(I1)
            V_predict = Phi(Ek_I1)
            V_predict = torch.reshape(V_predict, (V_predict.shape[0], 2, -1))
            
            positions_to_keypoint_representation = Positions_To_KeypointRepresentation()
            K_hat = positions_to_keypoint_representation(V_predict, points)
            K_hat = K_hat.unsqueeze(1)

            I1_hat = G(K_hat, Ea(I1))
            
            # K_hat = []
            # for batch_idx in range(K.shape[0]):
            #     out_img = positions_to_keypoint_representation(V_predict[batch_idx], points)
            #     K_hat.append(out_img)
            # K_hat = torch.stack(K_hat)
            # # K_hat = K_hat.unsqueeze(1)
            
            Recon_loss = ReconstructLoss(VGG_model)
            mse_loss = nn.MSELoss()
            L1_loss = nn.L1Loss()
            Adv_loss = AdversarialLoss(D)
            
            # Discriminator: max log(D(x)) + log(1 - D(G(z)))
            # disc_real = D(K).reshape(-1)
            # loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            # disc_Ek_I1 = D(Ek_I1).reshape(-1)
            # loss_disc_Ek_I1 = criterion(disc_Ek_I1, torch.zeros_like(disc_Ek_I1))
            # D_loss = (loss_disc_real + loss_disc_Ek_I1)
            D_loss = -1*Adv_loss(Ek_I1, K)
            
                  
            # landmarks encoder Ek, appearance encoder Ea, and the generator G
            rec_loss = Recon_loss(I1, I1_hat)
            cons_loss = L1_loss(Ek_I1, K_hat) # cons_loss = L1_loss(Ek_I1.reshape(K_hat.shape[0], -1), K_hat)
            # disc_Ek_I1 = D(Ek_I1).reshape(-1)
            # adv_loss = criterion(disc_Ek_I1, torch.ones_like(disc_Ek_I1))
            adv_loss = -1*torch.mean(D(Ek_I1))
            
            Ek_Ea_G_loss = rec_loss + (2*cons_loss) + (adv_weight*adv_loss)

            
            # regressor using consistent loss, regressor loss
            consistent_loss = L1_loss(K_hat, Ek_I1)
            # consistent_loss = L1_loss(K_hat, Ek_I1.detach().reshape(K_hat.shape[0], -1))
            regressor_loss = mse_loss(Phi(K), V_real)
            Phi_loss = consistent_loss + (0.5*regressor_loss)
            
            if running_idx % 20 == 0:

                fig, axs = plt.subplots(2, 3, figsize=(24, 12))

                I1_set = 255 * torch.permute(I1, (0, 2, 3, 1))            
                I1_img = np.uint8(I1_set[0].detach().cpu().numpy())
                axs[0, 0].imshow(I1_img, cmap='gray')
                axs[0, 0].set_title(r'$I_{1}$')

                K_set = 255 * torch.permute(Ek_I1, (0, 2, 3, 1))            
                K_img = np.uint8(K_set[0].detach().cpu().numpy())
                axs[0, 1].imshow(K_img, cmap='gray')
                axs[0, 1].set_title(r'$E_k(I_{1})$')

                K_hat_set = 255 * torch.permute(K_hat, (0, 2, 3, 1))    
                # K_hat_set = 255 * torch.permute(K_hat.reshape(K_hat.shape[0], 1, height, width), (0, 2, 3, 1)) 
                K_hat_img = np.uint8(K_hat_set[0].detach().cpu().numpy())
                axs[0, 2].imshow(K_hat_img, cmap='gray')
                axs[0, 2].set_title(r'$\hat{K}$')

                Ki_set = 255 * torch.permute(K, (0, 2, 3, 1))            
                Ki_img = np.uint8(Ki_set[0].detach().cpu().numpy())
                axs[1, 0].imshow(Ki_img, cmap='gray')
                axs[1, 0].set_title(r'$K_{real}$')

                Ea_set = 255 * torch.permute(Ea(I1), (0, 2, 3, 1))    
                Ea_img = np.uint8(Ea_set[0].detach().cpu().numpy())
                axs[1, 1].imshow(Ea_img, cmap='gray')
                axs[1, 1].set_title(r'$E_a(I_{1})$')

                I1_hat_set = 255 * torch.permute(I1_hat, (0, 2, 3, 1))            
                I1_hat_img = np.uint8(I1_hat_set[0].detach().cpu().numpy())
                axs[1, 2].imshow(I1_hat_img, cmap='gray')
                axs[1, 2].set_title(r'$\hat{I}_{1}$')

                fig.suptitle(f'Validation <-----> Epoch={epoch} and Iter={running_idx}')
                
                plt.show()
                            
                
            running_idx += 1 

            
            Ek_Ea_G_losses.append(Ek_Ea_G_loss.item())
            D_losses.append(D_loss.item())
            Phi_losses.append(Phi_loss.item())

            p_bar.set_description('Epoch {}'.format(epoch))
            p_bar.set_postfix(rec_loss=rec_loss.item(), D_loss=D_loss.item(), adv_loss=adv_loss.item(), 
                              reg_loss=regressor_loss.item(), consistent_loss=consistent_loss.item())

    print('Epoch: {} \t total_Ek_Eg_G_loss {:.4f}, \t total_D_loss {:.4f}, \t total_Phi_loss {:.4f}'.format(epoch, 
                                                                                                          np.mean(Ek_Ea_G_losses), 
                                                                                                          np.mean(D_losses), 
                                                                                                          np.mean(Phi_losses)))
    early_stopping(np.mean(Ek_Ea_G_losses), Ek, Ea, G, D, Phi, epoch)
    
    return np.mean(Ek_Ea_G_losses), np.mean(D_losses), np.mean(Phi_losses)


#%%
class ReconstructLoss(nn.Module):
    def __init__(self, f ):
        super(ReconstructLoss, self).__init__()
        self.f = f

    def forward(self, I1, I1_hat):
        M = I1.shape[0]

        # a1 = torch.sum(torch.norm(self.f(I1_hat.repeat(1, 3, 1, 1)) - self.f(I1.repeat(1, 3, 1, 1)), p=2, dim=(1,)))/(2*M)
        # a2 = torch.sum(torch.norm(I1_hat-I1, p=1, dim=(1,)))/(2*M)
        
        # a1 = torch.mean(torch.norm(self.f(I1_hat.repeat(1, 3, 1, 1)) - self.f(I1.repeat(1, 3, 1, 1)), p=2))
        rec_loss = torch.mean(torch.abs(I1_hat-I1))
                
        # rec_loss = (a1 + a2)/(2*M)
        
        # rec_loss = a1 + a2

        return rec_loss

#%%

class AdversarialLoss(nn.Module):
    def __init__(self, D):
        super(AdversarialLoss, self).__init__()
        self.D = D

    def forward(self, Ek_I1, K):
        a1 = torch.mean(self.D(K))
        a2 = torch.mean(self.D(Ek_I1))

        adv_loss = (a1 - a2)

        return adv_loss


#%%
# class ConsistenceLoss(nn.Module):
#     def __init__(self):
#         super(ConsistenceLoss, self).__init__()

#     def forward(self, V):
                
#         X = torch.unsqueeze(V[:, 0, :8], 2)
#         Y = torch.unsqueeze(V[:, 1, :8], 2)
                
#         A = torch.cat((X**2, X * Y, Y**2, X, Y), dim=2) 
#         A_inv = torch.pinverse(A)
#         Z = torch.bmm(A_inv, torch.ones_like(X))
#         pupil_residual = torch.sum((torch.bmm(A, Z) - torch.ones_like(X))**2, 1)
#         pupil_score_loss = torch.mean(pupil_residual)
                
        
#         X = torch.unsqueeze(V[:, 0, 8:], 2)
#         Y = torch.unsqueeze(V[:, 1, 8:], 2)
        
#         A = torch.cat((X**2, X * Y, Y**2, X, Y), dim=2) 
#         A_inv = torch.pinverse(A)
#         Z = torch.bmm(A_inv, torch.ones_like(X))
#         iris_residual = torch.sum((torch.bmm(A, Z) - torch.ones_like(X))**2, 1)
#         iris_score_loss = torch.mean(iris_residual)

#         con_loss = pupil_score_loss + iris_score_loss

#         return con_loss


#%%
class Positions_To_KeypointRepresentation(nn.Module):
    def __init__(self):
        super(Positions_To_KeypointRepresentation, self).__init__()

    def vectorized_point_to_line_dist(self, points, line0):
        points = points.repeat(line0.shape[0], line0.shape[1], 1, 1)
        line1 = torch.roll(line0, -1, 1)

        # Calculate the unit vector along the line segment
        unit_line = line1 - line0
        norm_unit_line = unit_line / torch.norm(unit_line, dim=2, keepdim=True)

        diff = (
            (norm_unit_line[:, :, 0].unsqueeze(-1) * (points[:, :, :, 0] - line0[:, :, 0].unsqueeze(-1))) +
            (norm_unit_line[:, :, 1].unsqueeze(-1) * (points[:, :, :, 1] - line0[:, :, 1].unsqueeze(-1)))
        )
        
        
        x_seg = (norm_unit_line[:, :, 0].unsqueeze(-1) * diff) + line0[:, :, 0].unsqueeze(-1)
        y_seg = (norm_unit_line[:, :, 1].unsqueeze(-1) * diff) + line0[:, :, 1].unsqueeze(-1)

        # Decide if the intersection point falls on the line segment
        lp1_x = line0[:, :, 0].unsqueeze(-1)
        lp1_y = line0[:, :, 1].unsqueeze(-1)
        lp2_x = line1[:, :, 0].unsqueeze(-1)
        lp2_y = line1[:, :, 1].unsqueeze(-1)

        is_betw_x = (lp1_x <= x_seg) & (x_seg <= lp2_x) | (lp2_x <= x_seg) & (x_seg <= lp1_x)
        is_betw_y = (lp1_y <= y_seg) & (y_seg <= lp2_y) | (lp2_y <= y_seg) & (y_seg <= lp1_y)

        condition = is_betw_x & is_betw_y
        
        
        # Compute the perpendicular distance to the theoretical infinite line
        point_line_vector = line0.unsqueeze(2) - points    
        segment_dist = torch.abs(
             ((unit_line[:, :, 0].unsqueeze(-1) * point_line_vector[:, :, :, 1]) -  (unit_line[:, :, 1].unsqueeze(-1) * point_line_vector[:, :, :, 0]))/
            torch.norm(unit_line, dim=2, keepdim=True)
        )

        segment_dist[~condition] = 100.0

        return segment_dist
    
    def forward(self, V_predict, points):
        
        X_values_pupil = V_predict[:, 0, :8]
        Y_values_pupil = V_predict[:, 1, :8] 
        
        positions_pupil = (torch.round(Y_values_pupil), torch.round(X_values_pupil))
        positions_pupil = torch.cat((positions_pupil[0].unsqueeze(2), positions_pupil[1].unsqueeze(2)), dim=2)
            
        X_values_iris = V_predict[:, 0, 8:]
        Y_values_iris = V_predict[:, 1, 8:]
        
        positions_iris = (torch.round(Y_values_iris), torch.round(X_values_iris))
        positions_iris = torch.cat((positions_iris[0].unsqueeze(2), positions_iris[1].unsqueeze(2)), dim=2)      
                    
        pos_flags_pupil = self.vectorized_point_to_line_dist(points, positions_pupil)
        min_pos_flags_pupil = torch.min(pos_flags_pupil, dim=1).values
        
        pos_flags_iris = self.vectorized_point_to_line_dist(points, positions_iris)
        min_pos_flags_iris = torch.min(pos_flags_iris, dim=1).values

        final_result = torch.minimum(min_pos_flags_pupil, min_pos_flags_iris)
        
        out_img_vector = torch.exp(-0.9 * final_result)
        
        out_image = out_img_vector.view(out_img_vector.shape[0], 70, 100)
        
        return out_image


#%%

