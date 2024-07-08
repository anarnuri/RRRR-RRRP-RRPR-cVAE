import torch
batch_size=2048
attention_dim=1024
n_attention_blocks=6
num_joints=8
n_heads=8
num_classes=3
beta_cvae=1
learning_rate=1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")