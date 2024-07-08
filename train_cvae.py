# export WANDB_API_KEY=PUT_YOUR_WANDB_API_KEY_HERE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset import SingleDataset
import torchvision.transforms as transforms
from attention_cvae import cVAE
from constants import *
import opendatasets as od 

od.download('https://www.kaggle.com/datasets/purwarlab/four-bar-coupler-curves')

torch.set_float32_matmul_precision('medium')
       
dataset = SingleDataset(transform=transforms.Compose([transforms.ToTensor(), 
                                                      transforms.Resize((32, 32), antialias=True), 
                                                      transforms.Lambda(lambda x: torch.flatten(x)), ]))

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, shuffle=True, num_workers=6, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_dataset, num_workers=6, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, num_workers=6, batch_size=batch_size, drop_last=True)

model = cVAE(learning_rate, attention_dim, num_joints, beta_cvae, n_attention_blocks, n_heads, batch_size)

checkpoint_callback = ModelCheckpoint(
    dirpath='weights/',
    monitor='val_loss',
    filename='{epoch}', 
    save_top_k=3)  

wandb_logger = WandbLogger(project='training_mixed_dataset')

if torch.cuda.device_count() == 1: 
    trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", max_epochs=-1, callbacks=[checkpoint_callback])
elif torch.cuda.device_count() > 1:
    trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", devices=-1, max_epochs=-1, strategy="ddp", callbacks=[checkpoint_callback])
else:
    trainer = pl.Trainer(logger=wandb_logger, accelerator="cpu", callbacks=[checkpoint_callback])
    
# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model
trainer.test(model, test_loader)