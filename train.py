import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle
# from config import Config
import config
from model import BiDAF
# from standardmodel import QANet
from utils import SQuADData
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import valid
import torch.optim as optim
from math import log2
from proc import load
from collections import Counter


# prepare data
print('prepare data')
# config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pre_trained_ = load('pre_data/embed_pre.json')
pre_trained = torch.Tensor(pre_trained_[0])
del pre_trained_
print('loading train_dataset')
train_dataset = SQuADData('pre_data/input/train')
dev_dataset = SQuADData('pre_data/input/dev')

# define model
print('define model')
model = BiDAF(pre_trained)
# model = torch.load('model/model.pt')
model = model.to(device)
lr = config.learning_rate
base_lr = 1.0
warm_up = config.lr_warm_up_num
cr = lr / log2(warm_up)
optimizer = torch.optim.Adam(lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps,
                             weight_decay=3e-7, params=model.parameters())
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * log2(ee + 1) if ee < warm_up else lr)

print('begin train')
for epoch in range(config.num_epoch):
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = iter(train_dataloader)
    losses = []
    model.train()
    for step in tqdm(range(len(train_dataset) // config.batch_size)):
        optimizer.zero_grad()
        cw, cc, qw, qc, y1s, y2s, ids = next(train_iter)
        cw, cc, qw, qc, y1s, y2s = cw.to(device), cc.to(device), qw.to(device), qc.to(device), y1s.to(device), y2s.to(device)
        p1, p2 = model(cw, cc)
        loss_1 = F.nll_loss(p1, y1s)
        loss_2 = F.nll_loss(p2, y2s)
        loss = (loss_1 + loss_2) / 2
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        if(step % 100 == 0):
            print('Epoch: %2d | Step: %3d | Loss: %3f' % (epoch, step, loss))
    torch.save(model, 'model/model.pt')
    f1, em, loss = valid(model, dev_dataset)
    print('-' * 30)
    print('Valid:')
    print('F1: %.2f | EM: %.2f | LOSS: %.2f' % (f1, em, loss))
