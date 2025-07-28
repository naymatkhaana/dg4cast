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
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        #print("x.shape::::::::::::::",x.shape)
        x = self.flatten(x)
        #print("x.shape::::::::::::::",x.shape)
        x = self.linear(x)
        x = self.dropout(x)
        return x

    
class CrossEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(CrossEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class CrossEncoderLayer(nn.Module):
    def __init__(self, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(CrossEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        #self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        #self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        #x = x + self.dropout(self.self_attention(
        #    x, x, x,
        #    attn_mask=x_mask,
        #    tau=tau, delta=None
        #)[0])
        #x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1) ####### fix it

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)
    



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


        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.ex_embedding_sh = DataEmbedding_inverted(self.seq_len_sh, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.ex_embedding_sh_pre = DataEmbedding_inverted(self.seq_len_sh_pre, configs.d_model, configs.embed, configs.freq, configs.dropout)


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


        # Encoder
        self.encoder_sh = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, n_heads_sh),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(e_layers_sh)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )


        # Encoder
        self.encoder_sh_pre = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, n_heads_sh_pre),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(e_layers_sh_pre)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

 
        
        ################################################################################################### cross 
        #self.cross_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,output_attention=False),configs.d_model, configs.n_heads)
        self.cross_encoder = CrossEncoder(
            [
                CrossEncoderLayer(
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                #for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.cross_encoder_sh = CrossEncoder(
            [
                CrossEncoderLayer(
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, n_heads_sh),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                #for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.cross_encoder_sh_pre = CrossEncoder(
            [
                CrossEncoderLayer(
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, n_heads_sh_pre),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                #for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        #self.cross_attention_sh = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,output_attention=False),configs.d_model, configs.n_heads_sh)
        #self.cross_attention_sh_pre = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,output_attention=False),configs.d_model, configs.n_heads_sh_pre)
        ###################################################################################################
         


        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)

        # Prediction Head
        self.head_nf_sh = configs.d_model * \
                       int((seq_len_sh - patch_len_sh) / stride_sh + 2)

        # Prediction Head
        self.head_nf_sh_pre = configs.d_model * \
                       int((seq_len_sh_pre - patch_len_sh_pre) / stride_sh_pre + 2)



        #print("self.head_nf:::::::::::",self.head_nf)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len-pred_len_sh-pred_len_sh_pre,
                                    head_dropout=configs.dropout)
            self.head_sh = FlattenHead(configs.enc_in, self.head_nf_sh, pred_len_sh,
                                    head_dropout=configs.dropout)
            self.head_sh_pre = FlattenHead(configs.enc_in, self.head_nf_sh_pre, pred_len_sh_pre,
                                    head_dropout=configs.dropout)

        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        #print("x_enc.shape:::", x_enc.shape)
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc_sh = x_enc[:,-4:,:].clone()
        x_enc_sh_pre = x_enc[:,-2:,:].clone()



        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        #print("enc_out.shape after patch embedding::::::::::::::",enc_out.shape)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, int(enc_out.shape[-2]), enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]



        ############ ******************************************************************************************** cross enc
        en_embed = enc_out[:,-1,:,:]
        ex_embed = self.ex_embedding(x_enc[:,:-1,:].permute(0, 2, 1),x_mark=None)
        #print("en_embed.shape:", en_embed.shape, "ex_embed.shape:",ex_embed.shape)
        res = self.cross_encoder(en_embed,ex_embed)
        
        res = res.permute(0, 2, 1).unsqueeze(1)
        #print("res.shape after permute::::::::::::::",res.shape)
        
        ############ ******************************************************************************************** cross enc




        enc_out = enc_out.permute(0, 1, 3, 2)
        #print("enc_out.shape::::::::::::::",enc_out.shape)

        ## Decoder
        #dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]


        # Decoder
        dec_out = self.head(res)  # z: [bs x nvars x target_window]
        dec_out = dec_out.repeat((1,self.enc_in, 1))
        #print("dec_out.shape:", dec_out.shape)



        dec_out = dec_out.permute(0, 2, 1)


        ####################################################################################################################

        # do patching and embedding on sh
        x_enc_sh = x_enc_sh.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        #print("x_enc.shape",x_enc.shape, "x_enc_sh.shape",x_enc_sh.shape)
        enc_out_sh, n_vars_sh = self.patch_embedding_sh(x_enc_sh)
        #print("enc_out_sh.shape after patch embedding::::::::::::::",enc_out_sh.shape)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out_sh, attns_sh = self.encoder_sh(enc_out_sh)
        # z: [bs x nvars x patch_num x d_model]
        enc_out_sh = torch.reshape(
            enc_out_sh, (-1, n_vars_sh, int(enc_out_sh.shape[-2]), enc_out_sh.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]


        ############ ******************************************************************************************** cross enc
        en_embed_sh = enc_out_sh[:,-1,:,:]
        ex_embed_sh = self.ex_embedding_sh(x_enc_sh[:,:-1,:].permute(0, 2, 1),x_mark=None)
        #print("en_embed.shape:", en_embed.shape, "ex_embed.shape:",ex_embed.shape)
        res_sh = self.cross_encoder_sh(en_embed_sh,ex_embed_sh)
        
        res_sh = res_sh.permute(0, 2, 1).unsqueeze(1)
        #print("res_sh.shape after permute::::::::::::::",res_sh.shape)
        
        ############ ******************************************************************************************** cross enc


        enc_out_sh = enc_out_sh.permute(0, 1, 3, 2)
        #print("enc_out_sh.shape::::::::::::::",enc_out_sh.shape)

        # Decoder
        #dec_out_sh = self.head_sh(enc_out_sh)  # z: [bs x nvars x target_window]

        # Decoder
        dec_out_sh = self.head_sh(res_sh)  # z: [bs x nvars x target_window]
        dec_out_sh = dec_out_sh.repeat((1,self.enc_in, 1))
        #print("dec_out.shape:", dec_out.shape)


        dec_out_sh = dec_out_sh.permute(0, 2, 1)

        #print("dec_out",dec_out.shape)
        #print("dec_out_sh",dec_out_sh.shape)



        ####################################################################################################################

        # do patching and embedding on sh
        x_enc_sh_pre = x_enc_sh_pre.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        #print("x_enc.shape",x_enc.shape, "x_enc_sh.shape",x_enc_sh.shape)
        enc_out_sh_pre, n_vars_sh_pre = self.patch_embedding_sh_pre(x_enc_sh_pre)
        #print("enc_out_sh.shape after patch embedding::::::::::::::",enc_out_sh.shape)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out_sh_pre, attns_sh_pre = self.encoder_sh_pre(enc_out_sh_pre)
        # z: [bs x nvars x patch_num x d_model]
        enc_out_sh_pre = torch.reshape(
            enc_out_sh_pre, (-1, n_vars_sh_pre, int(enc_out_sh_pre.shape[-2]), enc_out_sh_pre.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]



        ############ ******************************************************************************************** cross enc
        en_embed_sh_pre = enc_out_sh_pre[:,-1,:,:]
        ex_embed_sh_pre = self.ex_embedding_sh_pre(x_enc_sh_pre[:,:-1,:].permute(0, 2, 1),x_mark=None)
        #print("en_embed.shape:", en_embed.shape, "ex_embed.shape:",ex_embed.shape)
        res_sh_pre = self.cross_encoder_sh_pre(en_embed_sh_pre,ex_embed_sh_pre)
        
        res_sh_pre = res_sh_pre.permute(0, 2, 1).unsqueeze(1)
        #print("res_sh_pre.shape after permute::::::::::::::",res_sh_pre.shape)
        
        ############ ******************************************************************************************** cross enc


        enc_out_sh_pre = enc_out_sh_pre.permute(0, 1, 3, 2)
        #print("enc_out_sh.shape::::::::::::::",enc_out_sh.shape)

        ## Decoder
        #dec_out_sh_pre = self.head_sh_pre(enc_out_sh_pre)  # z: [bs x nvars x target_window]

        # Decoder
        dec_out_sh_pre = self.head_sh_pre(res_sh_pre)  # z: [bs x nvars x target_window]
        dec_out_sh_pre = dec_out_sh_pre.repeat((1,self.enc_in, 1))
        #print("dec_out.shape:", dec_out.shape)




        dec_out_sh_pre = dec_out_sh_pre.permute(0, 2, 1)


        dec_out = torch.cat((dec_out_sh_pre,dec_out_sh,dec_out), dim=1)


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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
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
