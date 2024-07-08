import torch
import torch.nn as nn
from torchmetrics import Accuracy
import pytorch_lightning as pl
from constants import *
from helpers import * 
import numpy as np 

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # Flatten the tensor (keep the batch size)
        return x.view(x.size(0), -1)
    
class UnFlatten(nn.Module):
    def __init__(self, h_dim):
        super(UnFlatten, self).__init__()
        self.h_dim = h_dim

    def forward(self, x):
        # Flatten the tensor (keep the batch size)
        w = 16
        return x.view(x.size(0), self.h_dim, w, w)

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
            
            self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)

class cVAE(pl.LightningModule):
    def __init__(self, learning_rate, encoder_dim, num_joints, beta, n_blocks, n_heads, batch_size):
        
        super().__init__()

        self.label_encode = nn.Linear(1, encoder_dim)
        # self.image_encode = nn.Linear(1024, encoder_dim)

        self.condition_cross_attention = CrossTransformerBlock(encoder_dim, n_heads, batch_size)
        
        self.joints_encode = nn.Linear(num_joints, encoder_dim)

        self.encoder_cross_attention = CrossTransformerBlock(encoder_dim, n_heads, batch_size)
        self.encoder_self_attentions = nn.ModuleList([SelfTransformerBlock(encoder_dim, n_heads, batch_size) for _ in range(n_blocks)])
                    
        self.calc_mean = MLP([encoder_dim, encoder_dim], last_activation=False)
        self.calc_logvar = MLP([encoder_dim, encoder_dim], last_activation=False)
               
        self.decoder_cross_attention = CrossTransformerBlock(encoder_dim, n_heads, batch_size)
        self.decoder_self_attentions = nn.ModuleList([SelfTransformerBlock(encoder_dim, n_heads, batch_size) for _ in range(n_blocks)])
        
        self.joint_predictor = nn.Linear(encoder_dim, num_joints)

        self.learning_rate = learning_rate
        self.beta = beta
        
    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)  
        z = mu + eps * std
        return z
    
    def forward(self, images, joints, labels):
        labels_encoded = self.label_encode(labels)
        # images_encoded = self.image_encode(images)

        conditions = self.condition_cross_attention(labels_encoded, images)               
        joints_encoded = self.joints_encode(joints)
        
        encoder_attention = self.encoder_cross_attention(joints_encoded, conditions)

        for encoder_self_attention in self.encoder_self_attentions:
            encoder_attention = encoder_self_attention(encoder_attention)

        mu = self.calc_mean(encoder_attention)
        logvar = self.calc_logvar(encoder_attention)
        z = self.reparametrize(mu, logvar)
        
        decoder_attention = self.decoder_cross_attention(z, conditions)

        for decoder_self_attention in self.decoder_self_attentions:
            decoder_attention = decoder_self_attention(decoder_attention)

        predicted_joints = self.joint_predictor(decoder_attention)
        
        return predicted_joints, mu, logvar
    
    def loss_fn(self, recon_x, x, mu, logvar):
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        mse_loss = nn.MSELoss(reduction='mean')(recon_x, x)
        
        return mse_loss + self.beta*KLD, mse_loss, KLD
  
    def training_step(self, batch, batch_idx):            
        latents, joints, labels = batch

        recon_joints, mu, logvar = self.forward(latents, joints, labels)

        loss, mse, kld = self.loss_fn(recon_joints, joints, mu, logvar)       

        self.log('train_loss', loss)
        self.log('mse_loss', mse)
        self.log('kld_loss', kld)
         
        return loss 
    
    def validation_step(self, batch, batch_idx):
        latents, joints, labels = batch
        recon_joints, mu, logvar = self.forward(latents, joints, labels)

        loss, mse, kld = self.loss_fn(recon_joints, joints, mu, logvar)
                                             
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_mse_loss', mse, sync_dist=True)
        self.log('val_kld_loss', kld, sync_dist=True)        
        
        return loss

    def test_step(self, batch, batch_idx):
        latents, joints, labels = batch
        recon_joints, mu, logvar = self.forward(latents, joints, labels)

        loss, mse, kld = self.loss_fn(recon_joints, joints, mu, logvar)
        
        return loss
    
    def on_epoch_end(self):
        val_losses = self.trainer.callback_metrics['val_mse_loss']
        self.log('val_mse_losses', val_losses, on_step=False, on_epoch=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.learning_rate))