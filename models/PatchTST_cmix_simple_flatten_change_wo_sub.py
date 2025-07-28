import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, DataEmbedding_inverted, PositionalEmbedding
import torch.nn.functional as F

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        #print("taregt_window::::::::::",target_window)
        #print("nf::::::::::",nf)
        self.linear =  nn.Linear(nf, target_window) #nn.Linear(nf*n_vars, target_window) # nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        #print("x.shape::::::::::::::",x.shape)
        x = self.flatten(x)
        #print("x.shape::::::::::::::",x.shape)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        patch_len = configs.patch_len
        stride = configs.stride
        padding = stride
        self.enc_in = configs.enc_in

        patch_len_sh_pre = 2
        stride_sh_pre = 1
        padding_sh_pre = 1
        n_heads_sh_pre = 2
        e_layers_sh_pre = 2
        seq_len_sh_pre = 2
        pred_len_sh_pre = 1


        patch_len_sh = 2
        stride_sh = 2
        padding_sh = 2
        n_heads_sh = 4
        e_layers_sh = 4
        seq_len_sh = 4
        pred_len_sh = 5  ### 3,4,5,6
        self.seq_len_sh =  seq_len_sh
        self.seq_len_sh_pre =  seq_len_sh_pre



        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # patching and embedding
        self.patch_embedding_sh = PatchEmbedding(
            configs.d_model, patch_len_sh, stride_sh, padding_sh, configs.dropout)

        # patching and embedding
        self.patch_embedding_sh_pre = PatchEmbedding(
            configs.d_model, patch_len_sh_pre, stride_sh_pre, padding_sh_pre, configs.dropout)


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )


        
        
        ################################################################################################### cross 
        ###################################################################################################
         
        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)


        #print("self.head_nf:::::::::::",self.head_nf)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_corr):

        #print("************************* x_corr.shape: ", x_corr.shape)
        #print("x_enc.shape:::", x_enc.shape)
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev


        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]

        #print("************************* x_enc.shape before patch embedding: ", x_enc.shape)
        enc_out, n_vars = self.patch_embedding(x_enc)
        #print("************************* enc_out.shape after patch embedding: ", enc_out.shape)
        n_patches = enc_out.shape[1]         

        #enc_out = enc_out.view(enc_out.shape[0]//n_vars,n_vars,enc_out.shape[1],enc_out.shape[2])
        #print("************************* enc_out.shape after view: ", enc_out.shape)

        # Flatten vars and patches -> (batch_size, n_vars * n_patches, dim)
        #enc_out = enc_out.reshape(enc_out.shape[0], enc_out.shape[1] * enc_out.shape[2], enc_out.shape[3])
        #print("************************* enc_out.shape after flattening: ", enc_out.shape)


        ########################## get corr_bias
        #with torch.no_grad():
        total_patches = n_vars * n_patches

        # Create variable index mapping for each patch
        var_indices = torch.arange(total_patches, device=x_corr.device) // n_patches  # (total_patches,)
    
        # Expand to get all pair combinations
        v_i = var_indices.unsqueeze(0).expand(total_patches, total_patches)
        v_j = var_indices.unsqueeze(1).expand(total_patches, total_patches)

        # Now index into corr_mi using advanced indexing
        corr_bias = x_corr[:, v_i, v_j]  # shape: (batch, total_patches, total_patches)

        #print("************************* corr_bias.shape: ", corr_bias.shape)
        ##print("************************* corr_bias: ", corr_bias)


        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        #print("************************* enc_out.shape before encoder: ", enc_out.shape)
        corr_bias = corr_bias*0
        enc_out, attns = self.encoder(enc_out)
        #print("************************* enc_out.shape after encoder: ", enc_out.shape)
        # z: [bs x nvars x patch_num x d_model]
        #print("************************* enc_out.shape before reshape: ", enc_out.shape)
        n_vars_flatten = 1 ## <= NOTE <=
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, int(enc_out.shape[-2]), enc_out.shape[-1]))
        #print("************************* enc_out.shape after reshape: ", enc_out.shape)

        # z: [bs x nvars x d_model x patch_num]
        #print("************************* enc_out.shape before permute: ", enc_out.shape)
        enc_out = enc_out.permute(0, 1, 3, 2)
        #print("************************* enc_out.shape after permute: ", enc_out.shape)

        ## Decoder
        #print("************************* enc_out.shape before head: ", enc_out.shape)
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        #print("************************* dec_out.shape after head: ", dec_out.shape)
        #print("************************* dec_out.shape before permute: ", dec_out.shape)
        dec_out = dec_out.permute(0, 2, 1)
        #print("************************* dec_out.shape after permute: ", dec_out.shape)

 

        #dec_out = torch.cat((dec_out), dim=1)


        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))



        ##############################################################################################################




        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, corr_mi, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, corr_mi) ## 
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
