import torch
import torch.nn as nn
import torch.nn.functional as F
# from config import Config
import config
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
import os
from math import fabs


# config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def f1_score(p1s, p2s, y1s, y2s):
    p1s = torch.argmax(p1s, dim=1).to(torch.float)
    p2s = torch.argmax(p2s, dim=1).to(torch.float)
    y1s = y1s.to(torch.float)
    y2s = y2s.to(torch.float)
    f1s = []
    for i in range(len(p1s)):
        predicted = -1
        if p1s[i] > y1s[i]:
            predicted = min(y2s[i] - p1s[i], p2s[i] - p1s[i])
            predicted = predicted + 1 if predicted >= 0 else 0
        else:
            predicted = min(p2s[i] - y1s[i], y2s[i] - y1s[i])
            predicted = predicted + 1 if predicted >= 0 else 0
        recall = predicted / (y2s[i] - y1s[i] + 1)
        precise = predicted / (torch.abs(p1s[i] - p2s[i]) + 1)
        if recall + precise == 0:
            f1s.append(0.)
        else:
            f1s.append((2*recall*precise / (recall + precise)).item())
    return sum(f1s) / len(f1s)


def em(p1s, p2s, y1s, y2s):
    p1s = torch.argmax(p1s, dim=1)
    p2s = torch.argmax(p2s, dim=1)
    ems = []
    for i in range(len(y1s)):
        ems.append(int(p1s[i] == y1s[i] and p2s[i] == y2s[i]))
    return sum(ems) / len(ems)


def valid(model, data):
    dataloader = DataLoader(data, config.val_batch_size, shuffle=False)
    dataiter = iter(dataloader)
    model.to(device)
    model.eval()
    losses = []
    f1s = []
    ems = []
    num_batch_nums = config.val_batch_size
    with torch.no_grad():
        for batch in range(100):
            cw, cc, qw, qc, y1s, y2s, ids = next(dataiter)
            cw, cc, qw, qc, y1s, y2s = cw.to(device), cc.to(device), qw.to(device), qc.to(device), y1s.to(
                device), y2s.to(device)
            p1s, p2s = model(cw, cc, qw, qc)
            print('P1:', torch.argmax(p1s, dim=-1), 'P2:', torch.argmax(p2s, dim=-1))
            print('y1:', y1s, 'y2:', y2s)
            print('-' * 50)
            loss_1 = F.nll_loss(p1s, y1s)
            loss_2 = F.nll_loss(p2s, y2s)
            loss = (loss_1 + loss_2) / 2
            losses.append(loss.item())
            f1s.append(f1_score(p1s, p2s, y1s, y2s))
            ems.append(em(p1s, p2s, y1s, y2s))
    return np.mean(f1s), np.mean(ems), np.mean(losses)


def get_val_data():
    data_dev = pickle.load(open('pre_data/data_dev.pkl', 'rb'))
    context_tokens = data_dev[0][0:2000]
    context_chars = data_dev[1][0:2000]
    question_tokens = data_dev[2][0:2000]
    question_chars = data_dev[3][0:2000]
    y1s = data_dev[4][0:2000]
    y2s = data_dev[5][0:2000]
    ids = data_dev[6][0:2000]
    pickle.dump((context_tokens, context_chars, question_tokens, question_chars,
                 y1s, y2s, ids), open('pre_data/data_val.pkl', 'wb'))



def get_input(input, word2idx, char2idx):
    context_tokens, context_chars, question_chars, question_tokens = [], [], [], []
    y1s, y2s = [], []
    ids = []
    for example in tqdm(input):
        context_token = get_word_idx(example['context_tokens'], word2idx, config.para_limit)
        question_token = get_word_idx(example['question_tokens'], word2idx, config.para_limit)
        context_char = get_char_idx(example['context_chars'], char2idx, config.para_limit, config.char_limit)
        question_char = get_char_idx(example['question_chars'], char2idx, config.para_limit, config.char_limit)
        y1 = example['y1s']
        y2 = example['y2s']
        id = example['uuid']
        context_tokens.append(context_token)
        question_tokens.append(question_token)
        context_chars.append(context_char)
        question_chars.append(question_char)
        y1s.append(y1)
        y2s.append(y2)
        ids.append(id)
    return context_tokens, context_chars, question_tokens, question_chars, y1s, y2s, ids


def get_word_idx(input, token2idx, limit):
    length = len(input)
    if(length <= limit):
        for i in range(limit - length):
            input.append('<pad>')
    else:
        input = input[:limit]
    result = [token2idx[x] if x in token2idx else 1 for x in input]
    return result


def get_char_idx(input, token2idx, para_limit, word_limit):
    para_lenght = len(input)
    pad = ['<pad>'] * word_limit
    if para_lenght <= para_limit:
        for i in range(para_limit - para_lenght):
            input.append(pad)
    else:
        input = input[:para_limit]
    for i in range(len(input)):
        if len(input[i]) <= word_limit:
            for _ in range(word_limit - len(input[i])):
                input[i].append('<pad>')
        else:
            input[i] = input[i][:word_limit]
    result = [[token2idx[x] if x in token2idx else 1 for x in word] for word in input]
    return result


class SQuADData(Dataset):
    def __init__(self, data_path):
        self.context_tokens = torch.load(os.path.join(data_path, 'context_tokens.pt'))
        self.context_chars = torch.load(os.path.join(data_path, 'context_chars.pt'))
        self.question_tokens = torch.load(os.path.join(data_path, 'question_tokens.pt'))
        self.question_chars = torch.load(os.path.join(data_path, 'question_chars.pt'))
        self.y1s = torch.load(os.path.join(data_path, 'y1s.pt'))
        self.y2s = torch.load(os.path.join(data_path, 'y2s.pt'))
        self.ids = torch.load(os.path.join(data_path, 'ids.pt'))

    def __len__(self):
        return len(self.y1s)

    def __getitem__(self, index):
        return self.context_tokens[index].long(),\
            self.context_chars[index].long(), \
            self.question_tokens[index].long(), \
            self.question_chars[index].long(), \
            self.y1s[index].long(), \
            self.y2s[index].long(), \
            self.ids[index].long()
