import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from models import PatchTST_cmix_simple_flatten_change_wo_sub # Replace with your PatchTST model import
import numpy as np
import argparse
import random
from data_provider.data_factory import data_provider

# Dummy data for illustration
def get_dummy_dataloader():
    x = torch.randn(128, 10, 96)  # (batch, variables, lookback_length)
    y = torch.randn(128, 10)      # Target
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def objective(trial):



    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str,  default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--with_retrain', type=int, default=0, help='status')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/illness/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='national_illness_24_4.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='ILITOTAL', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--num_train', type=int, default=914, help='status') #1025
    parser.add_argument('--num_test', type=int, default=228, help='status') #118

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--stride', type=int, default=8, help='stride')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False







    # Define search space
    args.patch_len = trial.suggest_int("patch_len", 8, 32)
    args.stride = trial.suggest_int("stride", 1, args.patch_len)
    args.d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    args.e_layers= trial.suggest_int("e_layers", 2, 6)
    args.heads = trial.suggest_categorical("heads", [4, 8])
    args.dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Define model
    model = PatchTST_cmix_simple_flatten_change_wo_sub.Model(args).float()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Data
    #train_loader = get_dummy_dataloader()
    train_data_set, train_loader = data_provider(args, "train")
   


    # Training loop
    model.train()
    for epoch in range(args.train_epochs):  # Keep short for tuning
        total_loss = 0.0
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_corr) in enumerate(train_loader):

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()

            optimizer.zero_grad()
            outputs = model(batch_x.float(), batch_x_mark, dec_inp, batch_y_mark, batch_corr)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:].float()
            batch_y = batch_y[:, -args.pred_len:, f_dim:].float()


            if 'PatchTST_sep' in args.model:
                loss = criterion(outputs[:,-args.pred_len+6:,:], batch_y[:,-args.pred_len+6:,:])
                loss_sh = criterion(outputs[:,-args.pred_len+1:-args.pred_len+6,:], batch_y[:,-args.pred_len+1:-args.pred_len+6,:])
                loss_sh_pre = criterion(outputs[:,-args.pred_len:-args.pred_len+1,:], batch_y[:,-args.pred_len:-args.pred_len+1,:])

                # Compute softmax-based weights
                loss_detach, loss_sh_detach, loss_sh_pre_detach = loss.detach(), loss_sh.detach() , loss_sh_pre.detach()  # Prevent gradient flow through weights

                w = torch.exp(-loss_detach) / (torch.exp(-loss_detach) + torch.exp(-loss_sh_detach)+ torch.exp(-loss_sh_pre_detach))

                w_sh = torch.exp(-loss_sh_detach) / (torch.exp(-loss_detach) + torch.exp(-loss_sh_detach)+ torch.exp(-loss_sh_pre_detach))

                w_sh_pre = torch.exp(-loss_sh_pre_detach) / (torch.exp(-loss_detach) + torch.exp(-loss_sh_detach)+ torch.exp(-loss_sh_pre_detach))

                        # Weighted loss
                loss = w * loss + w_sh * loss_sh + w_sh_pre * loss_sh_pre 

            else:
                loss = criterion(outputs, batch_y)


            #train_loss.append(loss.item())



            #xb, yb = xb.to( "cpu"), yb.to( "cpu")
            loss = loss.float()
            #out = model(xb)
            #loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Run tuning
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best trial:", study.best_trial.params)
