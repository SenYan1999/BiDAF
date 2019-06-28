import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

d_embed = 300
d_embed_char = 64
d_model = 128
p_dropout = 0.2

def mask_logits(x, x_mask):
    x_mask = x_mask.to(torch.float32)
    return x * x_mask + (1 - x_mask) * (-1e30)

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
        self.h = nn.Linear(d_embed + d_embed_char, d_model, bias=False)
        self.highway = nn.ModuleList([HighwayNet(d_model, d_model) for _ in range(2)])

    def forward(self, x):
        H = self.h(x)
        for highway in self.highway:
            H = highway(H)
        return H


class LSTMEncoder(nn.Module):
    def __init__(self, in_size, out_size, num_layers):
        super(LSTMEncoder, self).__init__()
        self.enc = nn.LSTM(input_size=in_size, hidden_size=out_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_dropout if num_layers > 1 else 0.)

    def forward(self, x, x_len: torch.Tensor):
        origin_len = x.shape[1]
        lengths, sorted_idx = x_len.sort(0, descending=True)
        x = x[sorted_idx]
        inp = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.enc(inp)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=origin_len)
        _, unsorted_idx = sorted_idx.sort(0)
        out = out[unsorted_idx]
        out = F.dropout(out, p_dropout)
        return out


class BiAttention(nn.Module):
    def __init__(self):
        super(BiAttention, self).__init__()
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, 2*d_model))
        self.bias = nn.Parameter(torch.zeros(1))
        self.c_weight = nn.Linear(2*d_model, 1, bias=False)
        self.q_weight = nn.Linear(2*d_model, 1, bias=False)
        for weight in (self.cq_weight, self.c_weight.weight, self.q_weight.weight):
            nn.init.xavier_uniform_(weight)

    def forward(self, c, q, c_mask, q_mask):
        c_len, q_len = c.shape[1], q.shape[1]
        c_mask, q_mask = c_mask.unsqueeze(1).expand(-1, q_len, -1), q_mask.unsqueeze(1).expand(-1, c_len, -1)
        c, q = F.dropout(c, p_dropout), F.dropout(q, p_dropout)
        s0, s1 = self.c_weight(c).expand(-1, -1, q_len), \
               self.q_weight(q).transpose(1, 2).expand(-1, c_len, -1)
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        S = s0 + s1 + s2 + self.bias

        S1 = F.softmax(mask_logits(S, q_mask), dim=-1)
        C2Q = torch.bmm(S1, q) # B * c_limit * 2d_model
        S2 = F.softmax(mask_logits(S.transpose(1, 2), c_mask), dim=-1)
        S2 = torch.bmm(S1, S2)
        Q2C = torch.bmm(S2, c) # B * c_limit * 2d_model

        return torch.cat((c, C2Q, c * C2Q, c * Q2C), dim=-1)


class BiDAF(nn.Module):
    def __init__(self, word_embed):
        super(BiDAF, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embed)
        self.word_embedding.weight[1].requires_grad = True
        self.char_embedding = nn.Embedding(96, d_embed_char, padding_idx=0)
        self.embed = Embedding()
        self.encoder = LSTMEncoder(d_model, d_model, 1)
        self.attention = BiAttention()
        self.model = LSTMEncoder(d_model * 8, d_model, 2)
        self.out = LSTMEncoder(d_model * 2, d_model ,2)
        self.start = nn.Linear(10 * d_model, 1)
        self.end = nn.Linear(10 * d_model, 1)

    def forward(self, cw, cc, qw, qc):
        cw_mask, qw_mask = torch.zeros_like(cw) != cw, torch.zeros_like(qw) != qw
        c_len, q_len = torch.sum(cw_mask, dim=-1), torch.sum(qw_mask, dim=-1)

        Cw, Qw = self.word_embedding(cw), self.word_embedding(qw)
        Cc, Qc = self.char_embedding(cc), self.char_embedding(qc)
        C, Q = torch.cat((Cw, torch.max(Cc, dim=-2)[0]), dim=-1), torch.cat((Qw, torch.max(Qc, dim=-2)[0]), dim=-1)
        C, Q = self.embed(C), self.embed(Q)

        C, Q = self.encoder(C, c_len), self.encoder(Q, q_len) # B * seq * 2d_model
        G = self.attention(C, Q, cw_mask, qw_mask) # B * seq * 8d_model
        M = self.model(G, c_len) # B * seq * 2d_model
        M_proj = self.out(M, c_len) # B * seq * 2d_model

        p1 = self.start(torch.cat((G, M), dim=-1)).squeeze()
        p2 = self.start(torch.cat((G, M_proj), dim=-1)).squeeze()

        p1, p2 = F.log_softmax(mask_logits(p1, cw_mask), dim=-1), F.log_softmax(mask_logits(p2, cw_mask), dim=-1)

        return p1, p2


