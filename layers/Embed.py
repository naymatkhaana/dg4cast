import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def compute_hist2d(x, y, bins):
    """
    Compute joint histogram of two variables.
    """
    joint_hist = torch.histc(x * bins, bins=bins, min=0, max=bins-1).unsqueeze(1) * \
                 torch.histc(y * bins, bins=bins, min=0, max=bins-1).unsqueeze(0)
    joint_hist /= joint_hist.sum()
    return joint_hist

def mutual_info_binned(x, y, bins=16, eps=1e-10):
    """
    Estimate mutual information between two 1D variables using histogram binning.
    """
    # Normalize inputs to [0, 1]
    x = (x - x.min()) / (x.max() - x.min() + eps)
    y = (y - y.min()) / (y.max() - y.min() + eps)
    
    # Digitize
    x_bin = torch.clamp((x * bins).long(), max=bins - 1)
    y_bin = torch.clamp((y * bins).long(), max=bins - 1)

    joint = torch.zeros((bins, bins), device=x.device)
    for i in range(len(x_bin)):
        joint[x_bin[i], y_bin[i]] += 1
    joint /= joint.sum()

    px = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)

    px_py = px @ py + eps
    nonzero = joint > 0
    mi = (joint[nonzero] * (joint[nonzero] / px_py[nonzero]).log()).sum()
    return mi

def compute_mi_matrix_pytorch(batch_tensor, bins=16):
    """
    Compute MI matrices for a batch of inputs.
    batch_tensor: (B, P, D) PyTorch tensor
    Returns: (B, P, P) MI matrix tensor (non-differentiable)
    """
    B, P, D = batch_tensor.shape
    result = torch.zeros((B, P, P), device=batch_tensor.device)

    for b in range(B):
        for i in range(P):
            for j in range(P):
                if i == j:
                    result[b, i, j] = 1.0
                else:
                    x = batch_tensor[b, i]
                    y = batch_tensor[b, j]
                    mi = mutual_info_binned(x, y, bins=bins)
                    result[b, i, j] = mi
    return result



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        #print("x.shape before patching:", x.shape)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        #print("x.shape after patching:", x.shape, "\n")
        #c = torch.reshape(x, (x.shape[0], x.shape[1]* x.shape[2], x.shape[3]))
        #x_corr = self.get_x_corr_mi(c)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


    def get_x_corr_mi(self, x):
        """
        x: Tensor of shape (B, P, D)
           B = batch size
           P = number of patches
           D = values in each patch
        Returns:
            Tensor of shape (B, P, P) with mutual information matrices (non-differentiable)
        """
        B, P, D = x.shape
        mi_matrices = []

        # Disable gradient tracking
        with torch.no_grad():
            for b in range(B):
                sample = x[b].cpu().numpy()  # (P, D)
                mi_matrix = np.zeros((P, P))

                for i in range(P):
                    for j in range(P):
                        if i == j:
                            mi_matrix[i, j] = 1.0
                        else:
                            # mutual_info_regression expects shape (n_samples, 1)
                            mi = mutual_info_regression(
                                sample[i].reshape(-1, 1),
                                sample[j],
                                discrete_features=False
                            )
                            mi_matrix[i, j] = mi[0]
                mi_matrices.append(mi_matrix)

        # Stack and return as torch tensor
        result = torch.tensor(mi_matrices, dtype=torch.float32)
        return result




class PatchEmbedding2(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding2, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        #print("x.shape before patching:", x.shape)
        x = torch.reshape(x, (x.shape[0] , x.shape[2]* x.shape[1], x.shape[3]))
        #print("x.shape after patching:", x.shape, "\n")

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
