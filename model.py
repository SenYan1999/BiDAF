import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

d_embed = 300
d_model = 128
p_dropout = 0.2

class HighwayNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(HighwayNet, self).__init__()
        self.g = nn.Linear(in_dim, out_dim)
        self.t = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        g = torch.sigmoid(self.g(x))
        t = torch.relu(self.t(x))
        t = F.dropout(t, p_dropout)
        return g*t + (1-g)*x


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.h = nn.Linear(d_embed, d_model, bias=False)
        self.highway = nn.ModuleList([HighwayNet(d_model, d_model) for _ in range(2)])

    def forward(self, x):
        H = self.h(x)
        for highway in self.highway:
            H = highway(H)
        return H


class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.enc = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x, x_len: torch.Tensor):
        origin_len = x.shape[0]
        lengths, sorted_idx = x_len.sort(0, descending=True)
        x = x[sorted_idx]
        input = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.enc(input)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=origin_len)
        _, unsorted_idx = sorted_idx.sort(0)
        out = out[unsorted_idx]
        return out


class BiAttention(nn.Module):
    def __init__(self):
        super(BiAttention, self).__init__()
        self.c_weight = nn.Parameter(torch.zeros(2*d_model, 1))
        self.q_weight = nn.Parameter(torch.zeros(2*d_model, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, 2*d_model))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q):
        c_len, q_len = c.shape[1], q.shape[1]
        c, q = F.dropout(c, p_dropout), F.dropout(q, p_dropout)
        s0, s1 = torch.matmul(c, self.c_weight).expand(-1, -1, q_len), \
               torch.matmul(c, self.q_weight).transpose(1, 2).expand(-1, c_len, -1)
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        

        return


class BiDAF(nn.Module):
    def __init__(self, word_embed):
        super(BiDAF, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embed)
        self.embed = Embedding()

    def forward(self, cw, qw):
        cw_mask, qw_mask = torch.zeros_like(cw) != cw, torch.zeros_like(qw) != qw
        c_len, q_len = torch.sum(cw_mask, dim=-1), torch.sum(dim=-1)

        w, q = self.word_embedding(cw), self.word_embedding(qw)
        w, q = self.embed(w), self.embed(q)

