import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
#from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
warnings.filterwarnings('ignore')
from data_provider.datam import get_series_and_dates
from sklearn.feature_selection import mutual_info_regression 
from scipy.stats import kendalltau
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
#import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.stride = args.stride
        self.patch_len = args.patch_len

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        #cols = [] #["icu_patients"] #,"hosp_patients"] #,"new_tests","OT"] #,people_vaccinated,people_fully_vaccinated,OT
        #cols = ["icu_patients","positive_rate"]
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * (1-0.1032) )
        num_test = int(len(df_raw) * 0.1032)
        
        num_train = self.args.num_train
        num_test = self.args.num_test

        num_train = int(len(df_raw) * (1-0.2) )
        num_test = int(len(df_raw) * 0.2)
        
        print("num_train:",num_train)
        print("num_test:",num_test)


        num_vali = len(df_raw) - num_train - num_test
        

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        print("border1s:",border1s )
        print("border2s:",border2s )

        print("df_raw.columns",df_raw.columns)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        """
        if self.features == 'M' or self.features == 'MS':
            print("df_raw.columns",df_raw.columns)
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        """

        if self.scale:
            print("df_data.columns", df_data.columns)
            print("df_data.values",df_data.values)
            print(np.where(df_data.values == '1/5/2009 0:00'))
            print("df_data.values[0]", df_data.values[0])

            print("df_data.values[73]", df_data.values[73])
            data = np.log1p(np.array(df_data.values,dtype="float64"))

            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        #print("################################# index: ", index)
        #print("####################################################### seq_x.shape:",seq_x.shape)

        ### patch
        """
        x = torch.unsqueeze(torch.tensor(seq_x),0)
        x = F.pad(x, (0,self.stride), mode='replicate')
        x = torch.transpose(x, 1, 2)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        #print("x.shape after patching:", x.shape, "\n")

        c = torch.reshape(x, (x.shape[0], x.shape[1]* x.shape[2], x.shape[3]))
        c = torch.transpose(c,1,2)
        c=torch.squeeze(c,0)
        c= np.array(c)
        corr_mi = self.__compute_mi_matrix__(c)
        """
        #if index == 2: 
            #print("########################################### seq_x:", seq_x,"########################################### data_x:", self.data_x)
        corr_mi = self.__compute_te_matrix__(seq_x) #, index)
        #self.__plot_map__(corr_mi,"./hmaps/"+str(self.set_type)+"_"+str(index)+"hmap.png")
        #print("####################################################### corr_mi.shape:",corr_mi.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, corr_mi

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def __plot_map__(self,data,fname):
    
       # Variable names for axis labels
       labels = list(np.arange(data.shape[0]))
       
       
       # Create the heatmap
       fig, ax = plt.subplots()
       cax = ax.imshow(data, cmap='coolwarm', interpolation='nearest')
       
       # Add colorbar
       fig.colorbar(cax)
       
       # Set axis ticks and labels
       ax.set_xticks(np.arange(len(labels)))
       ax.set_yticks(np.arange(len(labels)))
       ax.set_xticklabels(labels)
       ax.set_yticklabels(labels)
       
       # Rotate the x-axis labels for better readability
       plt.xticks(rotation=45)
       
       # Add values to each cell
       for i in range(len(data)):
           for j in range(len(data)):
               text = ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")
       
       #plt.title("Heatmap of 2D Matrix")
       plt.tight_layout()
       plt.savefig(fname)
       plt.close()
       
       #print("Heatmap saved as 'heatmap_matrix.png'")
       

    def __compute_corr_matrix__(self,inp):
        """
        Compute Pearson correlation matrix for a single sample.
        inp: torch.Tensor of shape (lookback_length, n_vars)
        Returns: torch.Tensor of shape (n_vars, n_vars)
        """
        # Convert to NumPy if necessary
        if isinstance(inp, torch.Tensor):
            inp = inp.detach().cpu().numpy()
    
        temp_scaler = StandardScaler()
        inp = temp_scaler.fit_transform(inp)

        # Transpose to shape (n_vars, lookback_length) for corrcoef
        inp = inp.T
        corr_matrix = np.corrcoef(inp)
    
        # Replace NaNs with 0 (in case of constant series)
        corr_matrix = np.nan_to_num(corr_matrix)
    
        return torch.tensor(corr_matrix, dtype=torch.float32)


    def __compute_kernel_granger_matrix__(self, inp, lag=1, alpha=1.0, gamma=1.0):
        """
        Compute Granger causality matrix using RBF kernel and Ridge regression.
        
        Parameters:
            inp: torch.Tensor or np.ndarray of shape (lookback_length, n_vars)
            lag: number of lags to use for autoregressive modeling
            alpha: regularization strength for Ridge regression
            gamma: RBF kernel width
    
        Returns:
            Granger causality matrix: np.ndarray of shape (n_vars, n_vars)
        """
        if hasattr(inp, 'numpy'):  # If it's a torch.Tensor
            sample = inp.numpy()
        else:
            sample = inp
        T, n_vars = sample.shape
        result = np.zeros((n_vars, n_vars))
    
        for i in range(n_vars):  # Target variable
            for j in range(n_vars):  # Candidate causal variable
                if i == j:
                    result[i, j] = 1.0
                    continue
                
                # Prepare lagged data
                X_self = []
                X_joint = []
                Y = []
                for t in range(lag, T):
                    past_i = sample[t - lag:t, i].flatten()
                    past_j = sample[t - lag:t, j].flatten()
                    X_self.append(past_i)
                    X_joint.append(np.concatenate([past_i, past_j]))
                    Y.append(sample[t, i])
                
                X_self = np.array(X_self)
                X_joint = np.array(X_joint)
                Y = np.array(Y)
                
                # Apply RBF kernel
                K_self = rbf_kernel(X_self, gamma=gamma)
                K_joint = rbf_kernel(X_joint, gamma=gamma)
                
                # Ridge regression to predict Y
                reg = Ridge(alpha=alpha)
                score_self = cross_val_score(reg, K_self, Y, scoring='r2', cv=3).mean()
                score_joint = cross_val_score(reg, K_joint, Y, scoring='r2', cv=3).mean()
                
                # Granger causality score = improvement in prediction
                gc_score = max(0, score_joint - score_self)
                result[i, j] = gc_score
        
        return result


    def compute_transfer_entropy(self,x, y, bins=2, lag=1): #bins=10,
        """
        Estimate transfer entropy TE_{X->Y} using histogram estimation.
        x, y: 1D numpy arrays
        lag: number of past steps used
        """
        # Prepare lagged series
        x_t = x[:-lag]
        y_t = y[:-lag]
        y_t1 = y[lag:]
    
        # Discretize
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        x_d = discretizer.fit_transform(x_t.reshape(-1, 1)).flatten().astype(int)
        y_d = discretizer.fit_transform(y_t.reshape(-1, 1)).flatten().astype(int)
        y1_d = discretizer.fit_transform(y_t1.reshape(-1, 1)).flatten().astype(int)
    
        # Joint and marginal distributions
        p_xyz, _ = np.histogramdd(np.stack([x_d, y_d, y1_d], axis=1), bins=bins)
        p_xyz = p_xyz / np.sum(p_xyz)
    
        p_yz, _ = np.histogramdd(np.stack([y_d, y1_d], axis=1), bins=bins)
        p_yz = p_yz / np.sum(p_yz)
    
        p_y, _ = np.histogram(y_d, bins=bins)
        p_y = p_y / np.sum(p_y)
    
        p_z, _ = np.histogram(y1_d, bins=bins)
        p_z = p_z / np.sum(p_z)
    
        te = 0.0
        for i in range(bins):
            for j in range(bins):
                for k in range(bins):
                    p_ijk = p_xyz[i, j, k]
                    if p_ijk > 0:
                        p_jk = p_yz[j, k]
                        p_j = p_y[j]
                        p_k = p_z[k]
                        if p_jk > 0 and p_j > 0 and p_k > 0:
                            te += p_ijk * np.log((p_ijk * p_j) / (p_jk * p_xyz[i, :, :].sum()))
    
        return te
    
    def __compute_te_matrix__(self, inp, bins=10, lag=1):# bins=10
        """
        Compute pairwise transfer entropy TE_{i->j} matrix.
        inp: torch.Tensor of shape (lookback_length, n_vars)
        Returns: np.ndarray of shape (n_vars, n_vars)
        """
        if hasattr(inp, 'detach'):
            inp = inp.detach().cpu().numpy()
    
        _, n_vars = inp.shape
        result = np.zeros((n_vars, n_vars))
    
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    result[i, j] = 0.0  # No self-transfer
                else:
                    result[i, j] = self.compute_transfer_entropy(inp[:, i], inp[:, j], bins=bins, lag=lag)
    
        return result # 0.1* result
    
    
    def __compute_kendall_tau_matrix__(self, inp):
        """
        Compute Kendall's tau correlation matrix for each sample in the batch.
        inp: torch.Tensor or np.ndarray of shape (lookback_length, n_vars)
        Returns: np.ndarray of shape (n_vars, n_vars)
        """
        # Convert to numpy if input is torch tensor
        if hasattr(inp, 'detach'):
            inp = inp.detach().cpu().numpy()
    
        _, n_vars = inp.shape
        result = np.zeros((n_vars, n_vars))
    
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    result[i, j] = 1.0
                else:
                    tau, _ = kendalltau(inp[:, i], inp[:, j],nan_policy="omit")
                    result[i, j] =  tau if tau is not None and not np.isnan(tau) else 0

                    #print(i, j, result[i,j])
    
        return result # -0.1*result

    def __compute_mi_matrix__(self,inp, ind = 0):
        """
        Compute mutual information matrices for each sample in the batch.
        inp: torch.Tensor of shape ( lookback_length, n_vars)
        Returns: torch.Tensor of shape (n_vars, n_vars)
        """

        #if ind == 31:	print("##################33 inp:", inp)
        _, n_vars = inp.shape
        result = np.zeros((n_vars, n_vars))

        sample = inp
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    result[ i, j] = 1.0
                else:
                    try:
                        result[ i, j] = mutual_info_regression(sample[:,i].reshape(-1, 1), sample[:,j])[0]
                    except:
                        we = 32#print(sample[:,i].reshape(-1, 1), sample[:,j])
                        #print("##################33 inp:", inp, "##### ind: ", ind)

        return result # -0.01*result



    def __compute_distance_corr_matrix__(self, inp):
        """
        Compute distance correlation matrix for each sample in the batch.
        inp: torch.Tensor or np.ndarray of shape (lookback_length, n_vars)
        Returns: np.ndarray of shape (n_vars, n_vars)
        """
        if hasattr(inp, 'detach'):
            inp = inp.detach().cpu().numpy()
    
        def distance_correlation(x, y):
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            n = x.shape[0]
    
            # Compute pairwise Euclidean distances
            a = np.abs(x - x.T)
            b = np.abs(y - y.T)
    
            # Double center the distance matrices
            A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
            # Compute distance covariance and variances
            dcov2 = np.sum(A * B) / (n * n)
            dvar_x = np.sum(A * A) / (n * n)
            dvar_y = np.sum(B * B) / (n * n)
    
            if dvar_x == 0 or dvar_y == 0:
                return 0.0
    
            return np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y))
    
        _, n_vars = inp.shape
        result = np.zeros((n_vars, n_vars))
    
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    result[i, j] = 1.0
                else:
                    result[i, j] = distance_correlation(inp[:, i], inp[:, j])
    
        return result # -0.01*result


import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from bisect import bisect_right

class Dataset_Custom_multi(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S',data_path='ETTh1.csv', data_files=None, target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_files = data_files if data_files is not None else []
        if flag == 'test':
            self.data_files = ['national_illness_24_4_2cols.csv']  #['national_illness_24_4.csv']
        else:
            #self.data_files = ['ILI_mi.csv','ILI_ga.csv','ILI_ks.csv'] #,'national_illness_24.csv']
            self.data_files = ['ILI_mo.csv','ILI_pa.csv' ,'ILI_nv.csv' ,'ILI_ma.csv' ,'ILI_nd.csv' ,'ILI_nc.csv' ,'ILI_ia.csv' ,'ILI_ms.csv' ,'ILI_sc.csv' ,'ILI_nm.csv' ,'ILI_nj.csv' ,'ILI_mt.csv' ,'ILI_wv.csv' ,'ILI_sd.csv' ,'ILI_ky.csv' ,'ILI_ok.csv' ,'ILI_az.csv' ,'ILI_hi.csv' ,'ILI_nyc.csv' ,'ILI_tn.csv' ,'ILI_in.csv' ,'ILI_pr.csv' ,'ILI_ne.csv' ,'ILI_wy.csv' ,'ILI_ny.csv' ,'ILI_mi.csv' ,'ILI_mn.csv' ,'ILI_ut.csv' ,'ILI_la.csv'  ,'ILI_ar.csv'  ,'ILI_vi.csv'  ,'ILI_or.csv'  ,'ILI_ri.csv'  ,'ILI_nh.csv'  ,'ILI_id.csv'  ,'ILI_de.csv'  ,'ILI_tx.csv'  ,'ILI_md.csv'  ,'ILI_wa.csv'  ,'ILI_oh.csv'  ,'ILI_vt.csv'  ,'ILI_ks.csv'  ,'ILI_va.csv'  ,'ILI_al.csv'  ,'ILI_ak.csv'  ,'ILI_ct.csv'  ,'ILI_wi.csv'  ,'ILI_ga.csv'  ,'ILI_me.csv'  ,'ILI_il.csv'  ,'ILI_ca.csv'  ,'ILI_co.csv'] #,'national_illness_24.csv']

        self.data = []  # Stores processed data from all files
        self.cumulative_lengths = []  # Cumulative lengths for indexing across multiple files

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        cumulative_sum = 0

        for data_file in self.data_files:
            df_raw = pd.read_csv(os.path.join(self.root_path, data_file))

            # Ensure columns are properly ordered
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]

            num_train = int(len(df_raw) * (1 - 0.1032))
            num_test = int(len(df_raw) * 0.1032)
            num_train = int(len(df_raw) * (1 - 0.2))
            num_test = int(len(df_raw) * 0.2)

            num_vali = len(df_raw) - num_train - num_test
            print("num_train:",num_train)
            print("num_test:",num_test)

            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            print("border1s:",border1s )
            print("border2s:",border2s )

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features in ['M', 'MS']:
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            if self.scale:
                data = np.log1p(df_data.values)
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)

            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            series_data = {
                'data_x': data[border1:border2],
                'data_y': data[border1:border2],
                'data_stamp': data_stamp
            }

            self.data.append(series_data)
            series_length = len(series_data['data_x']) - self.seq_len - self.pred_len + 1
            cumulative_sum += series_length
            self.cumulative_lengths.append(cumulative_sum)
            print("self.cumulative_lengths", self.cumulative_lengths)

    def __getitem__(self, index):
        #print("***************************************** index: ", index)
        if index >= self.cumulative_lengths[-1]:
            raise IndexError("Index out of range")

        # Use binary search to find the appropriate time series
        dataset_index = bisect_right(self.cumulative_lengths, index)
        if dataset_index > 0:
            index -= self.cumulative_lengths[dataset_index - 1]

        series = self.data[dataset_index]
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = series['data_x'][s_begin:s_end]
        seq_y = series['data_y'][r_begin:r_end]
        seq_x_mark = series['data_stamp'][s_begin:s_end]
        seq_y_mark = series['data_stamp'][r_begin:r_end]

        corr_mi = self.__compute_te_matrix__(seq_x)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, corr_mi #np.zeros((seq_x.shape[1], seq_x.shape[1]))

    def __len__(self):
        return self.cumulative_lengths[-1]


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


    def __compute_corr_matrix__(self,inp):
        """
        Compute Pearson correlation matrix for a single sample.
        inp: torch.Tensor of shape (lookback_length, n_vars)
        Returns: torch.Tensor of shape (n_vars, n_vars)
        """
        # Convert to NumPy if necessary
        if isinstance(inp, torch.Tensor):
            inp = inp.detach().cpu().numpy()
    
        temp_scaler = StandardScaler()
        inp = temp_scaler.fit_transform(inp)

        # Transpose to shape (n_vars, lookback_length) for corrcoef
        inp = inp.T
        corr_matrix = np.corrcoef(inp)
    
        # Replace NaNs with 0 (in case of constant series)
        corr_matrix = np.nan_to_num(corr_matrix)
    
        return torch.tensor(corr_matrix, dtype=torch.float32)


    def __compute_kernel_granger_matrix__(self, inp, lag=1, alpha=1.0, gamma=1.0):
        """
        Compute Granger causality matrix using RBF kernel and Ridge regression.
        
        Parameters:
            inp: torch.Tensor or np.ndarray of shape (lookback_length, n_vars)
            lag: number of lags to use for autoregressive modeling
            alpha: regularization strength for Ridge regression
            gamma: RBF kernel width
    
        Returns:
            Granger causality matrix: np.ndarray of shape (n_vars, n_vars)
        """
        if hasattr(inp, 'numpy'):  # If it's a torch.Tensor
            sample = inp.numpy()
        else:
            sample = inp
        T, n_vars = sample.shape
        result = np.zeros((n_vars, n_vars))
    
        for i in range(n_vars):  # Target variable
            for j in range(n_vars):  # Candidate causal variable
                if i == j:
                    result[i, j] = 1.0
                    continue
                
                # Prepare lagged data
                X_self = []
                X_joint = []
                Y = []
                for t in range(lag, T):
                    past_i = sample[t - lag:t, i].flatten()
                    past_j = sample[t - lag:t, j].flatten()
                    X_self.append(past_i)
                    X_joint.append(np.concatenate([past_i, past_j]))
                    Y.append(sample[t, i])
                
                X_self = np.array(X_self)
                X_joint = np.array(X_joint)
                Y = np.array(Y)
                
                # Apply RBF kernel
                K_self = rbf_kernel(X_self, gamma=gamma)
                K_joint = rbf_kernel(X_joint, gamma=gamma)
                
                # Ridge regression to predict Y
                reg = Ridge(alpha=alpha)
                score_self = cross_val_score(reg, K_self, Y, scoring='r2', cv=3).mean()
                score_joint = cross_val_score(reg, K_joint, Y, scoring='r2', cv=3).mean()
                
                # Granger causality score = improvement in prediction
                gc_score = max(0, score_joint - score_self)
                result[i, j] = gc_score
        
        return result


    def compute_transfer_entropy(self,x, y, bins=2, lag=1): #bins=10,
        """
        Estimate transfer entropy TE_{X->Y} using histogram estimation.
        x, y: 1D numpy arrays
        lag: number of past steps used
        """
        # Prepare lagged series
        x_t = x[:-lag]
        y_t = y[:-lag]
        y_t1 = y[lag:]
    
        # Discretize
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        x_d = discretizer.fit_transform(x_t.reshape(-1, 1)).flatten().astype(int)
        y_d = discretizer.fit_transform(y_t.reshape(-1, 1)).flatten().astype(int)
        y1_d = discretizer.fit_transform(y_t1.reshape(-1, 1)).flatten().astype(int)
    
        # Joint and marginal distributions
        p_xyz, _ = np.histogramdd(np.stack([x_d, y_d, y1_d], axis=1), bins=bins)
        p_xyz = p_xyz / np.sum(p_xyz)
    
        p_yz, _ = np.histogramdd(np.stack([y_d, y1_d], axis=1), bins=bins)
        p_yz = p_yz / np.sum(p_yz)
    
        p_y, _ = np.histogram(y_d, bins=bins)
        p_y = p_y / np.sum(p_y)
    
        p_z, _ = np.histogram(y1_d, bins=bins)
        p_z = p_z / np.sum(p_z)
    
        te = 0.0
        for i in range(bins):
            for j in range(bins):
                for k in range(bins):
                    p_ijk = p_xyz[i, j, k]
                    if p_ijk > 0:
                        p_jk = p_yz[j, k]
                        p_j = p_y[j]
                        p_k = p_z[k]
                        if p_jk > 0 and p_j > 0 and p_k > 0:
                            te += p_ijk * np.log((p_ijk * p_j) / (p_jk * p_xyz[i, :, :].sum()))
    
        return te
    
    def __compute_te_matrix__(self, inp, bins=10, lag=1):# bins=10
        """
        Compute pairwise transfer entropy TE_{i->j} matrix.
        inp: torch.Tensor of shape (lookback_length, n_vars)
        Returns: np.ndarray of shape (n_vars, n_vars)
        """
        if hasattr(inp, 'detach'):
            inp = inp.detach().cpu().numpy()
    
        _, n_vars = inp.shape
        result = np.zeros((n_vars, n_vars))
    
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    result[i, j] = 0.0  # No self-transfer
                else:
                    result[i, j] = self.compute_transfer_entropy(inp[:, i], inp[:, j], bins=bins, lag=lag)
    
        return result #0.1* result
    
    
    def __compute_kendall_tau_matrix__(self, inp):
        """
        Compute Kendall's tau correlation matrix for each sample in the batch.
        inp: torch.Tensor or np.ndarray of shape (lookback_length, n_vars)
        Returns: np.ndarray of shape (n_vars, n_vars)
        """
        # Convert to numpy if input is torch tensor
        if hasattr(inp, 'detach'):
            inp = inp.detach().cpu().numpy()
    
        _, n_vars = inp.shape
        result = np.zeros((n_vars, n_vars))
    
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    result[i, j] = 1.0
                else:
                    tau, _ = kendalltau(inp[:, i], inp[:, j],nan_policy="omit")
                    result[i, j] =  tau if tau is not None and not np.isnan(tau) else 0

                    #print(i, j, result[i,j])
    
        return result # -0.1*result


    def __compute_distance_corr_matrix__(self, inp):
        """
        Compute distance correlation matrix for each sample in the batch.
        inp: torch.Tensor or np.ndarray of shape (lookback_length, n_vars)
        Returns: np.ndarray of shape (n_vars, n_vars)
        """
        if hasattr(inp, 'detach'):
            inp = inp.detach().cpu().numpy()
    
        def distance_correlation(x, y):
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            n = x.shape[0]
    
            # Compute pairwise Euclidean distances
            a = np.abs(x - x.T)
            b = np.abs(y - y.T)
    
            # Double center the distance matrices
            A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
            # Compute distance covariance and variances
            dcov2 = np.sum(A * B) / (n * n)
            dvar_x = np.sum(A * A) / (n * n)
            dvar_y = np.sum(B * B) / (n * n)
    
            if dvar_x == 0 or dvar_y == 0:
                return 0.0
    
            return np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y))
    
        _, n_vars = inp.shape
        result = np.zeros((n_vars, n_vars))
    
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    result[i, j] = 1.0
                else:
                    result[i, j] = distance_correlation(inp[:, i], inp[:, j])
    
        return result #-0.01*result

    def __compute_mi_matrix__(self,inp, ind = 0):
        """
        Compute mutual information matrices for each sample in the batch.
        inp: torch.Tensor of shape ( lookback_length, n_vars)
        Returns: torch.Tensor of shape (n_vars, n_vars)
        """

        #if ind == 31:	print("##################33 inp:", inp)
        _, n_vars = inp.shape
        result = np.zeros((n_vars, n_vars))

        sample = inp
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    result[ i, j] = 1.0
                else:
                    try:
                        result[ i, j] = mutual_info_regression(sample[:,i].reshape(-1, 1), sample[:,j])[0]
                    except:
                        we = 32#print(sample[:,i].reshape(-1, 1), sample[:,j])
                        #print("##################33 inp:", inp, "##### ind: ", ind)

        return result


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class Dataset_datam(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data_files = [ 'electricity_weekly_dataset'] # 'electricity_weekly_dataset']#,  'hospital_dataset',  'kdd_cup_2018_dataset_without_missing_values',    'm1_monthly_dataset'] #,  'm1_quarterly_dataset',  'm1_yearly_dataset',  'm3_monthly_dataset', 'm3_quarterly_dataset',  'm3_yearly_dataset',  'm4_daily_dataset',  'm4_hourly_dataset',  'm4_monthly_dataset',  'm4_quarterly_dataset',  'm4_weekly_dataset',   'pedestrian_counts_dataset',  'rideshare_dataset_without_missing_values',  'saugeenday_dataset',  'solar_10_minutes_dataset',  'solar_4_seconds_dataset',  'solar_weekly_dataset',  'sunspot_dataset_without_missing_values',  'temperature_rain_dataset_without_missing_values',  'tourism_monthly_dataset',  'tourism_quarterly_dataset',  'tourism_yearly_dataset',  'traffic_hourly_dataset',  'traffic_weekly_dataset',  'us_births_dataset',  'vehicle_trips_dataset_without_missing_values', 'wind_4_seconds_dataset',  'wind_farms_minutely_dataset_without_missing_values']   

       # self.data_files = [  'electricity_weekly_dataset',  'fred_md_dataset',  'hospital_dataset',  'kdd_cup_2018_dataset_without_missing_values',    'm1_monthly_dataset',  'm1_quarterly_dataset',  'm1_yearly_dataset',  'm3_monthly_dataset', 'm3_quarterly_dataset',  'm3_yearly_dataset',  'm4_daily_dataset',  'm4_hourly_dataset',  'm4_monthly_dataset',  'm4_quarterly_dataset',  'm4_weekly_dataset',   'pedestrian_counts_dataset',  'rideshare_dataset_without_missing_values',  'saugeenday_dataset',  'solar_10_minutes_dataset',  'solar_4_seconds_dataset',  'solar_weekly_dataset',  'sunspot_dataset_without_missing_values',  'temperature_rain_dataset_without_missing_values',  'tourism_monthly_dataset',  'tourism_quarterly_dataset',  'tourism_yearly_dataset',  'traffic_hourly_dataset',  'traffic_weekly_dataset',  'us_births_dataset',  'vehicle_trips_dataset_without_missing_values', 'wind_4_seconds_dataset',  'wind_farms_minutely_dataset_without_missing_values']   
        #
        #'car_parts_dataset_without_missing_values',  
        #, 'kaggle_web_traffic_dataset_without_missing_values'
        #'london_smart_meters_dataset_without_missing_values',
        # '  'm3_other_dataset', '
        #'m4_yearly_dataset', 
        #' 'weather_dataset','

        self.__read_data__()


    def __read_data__(self):
        self.data_x = []; self.data_y = []; self.data_stamp = []; self.data_lengths = []
        
        if self.set_type == 2 or self.set_type == 1:
            df_read = pd.read_csv(os.path.join(self.root_path,self.data_path))
            self.__add_data__(df_read) 
            self.dlengths = np.array(self.data_lengths) - self.seq_len - self.pred_len + 1
            self.cumsum = np.cumsum(self.dlengths)
            return

        for dfile in self.data_files:
            print("************* dfile: ", dfile)
            diter = 0
            for dseries, dates in get_series_and_dates("dataset/datam/"+dfile+"/"+dfile+".tsf"):
                print(diter); diter+=1;
                df_read = pd.DataFrame()
                df_read[self.target] = dseries
                df_read['date'] = dates
            
                self.__add_data__(df_read)  
            print("############### len(self.data_lengths): ",len(self.data_lengths))    
            print("############### np.sum(np.array(self.data_lengths)): ",np.sum(np.array(self.data_lengths)) )   
            print("############### np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1): ", np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1) )      

        print("############### Total len(self.data_lengths): ",len(self.data_lengths))    
        print("############### Total np.sum(np.array(self.data_lengths)): ",np.sum(np.array(self.data_lengths)) )   
        print("############### Total np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1): ", np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1) )      
        self.dlengths = np.array(self.data_lengths) - self.seq_len - self.pred_len + 1
        self.cumsum = np.cumsum(self.dlengths)


    def __add_data__(self,df_raw):
        self.scaler = StandardScaler()
        #df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * (1 if self.set_type == 0 else 0.7))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x.append(data[border1:border2])
        self.data_y.append(data[border1:border2])
        self.data_lengths.append(len(data[border1:border2]))

        #if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp.append(data_stamp)

    def __getitem__(self, index):

        

        #it = 0
        #ind = index
        #cumsum = dlengths[it]
        
        #while cumsum <= ind:
        #    it+=1
        #    cumsum += dlengths[it]
            
        # example dlengths: [6, 4, 6, 3, 3, 5]    , ind: 10
        
        # Calculate the cumulative sum of lengths
    
        it = np.searchsorted(self.cumsum, index, side='right')
    
        s_begin = self.dlengths[it] - (self.cumsum[it] - index)

        #s_begin =  self.dlengths[it] - (self.cumsum[it]-index) #index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[it][s_begin:s_end]
        seq_y = self.data_y[it][r_begin:r_end]
        seq_x_mark = self.data_stamp[it][s_begin:s_end]
        seq_y_mark = self.data_stamp[it][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1)

        #return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)










class Dataset_datam_short(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data_files = [  'electricity_weekly_dataset',  'fred_md_dataset',  'hospital_dataset',  'kdd_cup_2018_dataset_without_missing_values',    'm1_monthly_dataset',  'm1_quarterly_dataset',  'm1_yearly_dataset',  'm3_monthly_dataset', 'm3_quarterly_dataset',  'm3_yearly_dataset',  'm4_daily_dataset',  'm4_hourly_dataset',  'm4_monthly_dataset',  'm4_quarterly_dataset',  'm4_weekly_dataset',   'pedestrian_counts_dataset',  'rideshare_dataset_without_missing_values',  'saugeenday_dataset',  'solar_10_minutes_dataset',  'solar_4_seconds_dataset',  'solar_weekly_dataset',  'sunspot_dataset_without_missing_values',  'temperature_rain_dataset_without_missing_values',  'tourism_monthly_dataset',  'tourism_quarterly_dataset',  'tourism_yearly_dataset',  'traffic_hourly_dataset',  'traffic_weekly_dataset',  'us_births_dataset',  'vehicle_trips_dataset_without_missing_values', 'wind_4_seconds_dataset',  'wind_farms_minutely_dataset_without_missing_values']   
        #
        #'car_parts_dataset_without_missing_values',  
        #, 'kaggle_web_traffic_dataset_without_missing_values'
        #'london_smart_meters_dataset_without_missing_values',
        # '  'm3_other_dataset', '
        #'m4_yearly_dataset', 
        #' 'weather_dataset','

        self.__read_data__()


    def __read_data__(self):
        self.data_x = []; self.data_y = []; self.data_stamp = []; self.data_lengths = []
        
        if self.set_type == 2 or self.set_type == 1:
            df_read = pd.read_csv(os.path.join(self.root_path,self.data_path))
            self.__add_data__(df_read) 
            self.dlengths = np.array(self.data_lengths) - self.seq_len - self.pred_len + 1
            self.cumsum = np.cumsum(self.dlengths)
            return

        for dfile in self.data_files:
            print("************* dfile: ", dfile)
            diter = 0
            for dseries, dates in get_series_and_dates("dataset/datam/"+dfile+"/"+dfile+".tsf"):
                print(diter); diter+=1;
                df_read = pd.DataFrame()
                df_read['OT'] = dseries
                df_read['date'] = dates
            
                self.__add_data__(df_read)  
            print("############### len(self.data_lengths): ",len(self.data_lengths))    
            print("############### np.sum(np.array(self.data_lengths)): ",np.sum(np.array(self.data_lengths)) )   
            print("############### np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1): ", np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1) )      

        print("############### Total len(self.data_lengths): ",len(self.data_lengths))    
        print("############### Total np.sum(np.array(self.data_lengths)): ",np.sum(np.array(self.data_lengths)) )   
        print("############### Total np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1): ", np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1) )      
        self.dlengths = np.array(self.data_lengths) - self.seq_len - self.pred_len + 1
        self.cumsum = np.cumsum(self.dlengths)
        self.total_len = np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1)
        self.sample_arr = np.random.randint(self.total_len, size=(int(self.total_len*0.01),))



    def __add_data__(self,df_raw):
        self.scaler = StandardScaler()
        #df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * (1 if self.set_type == 0 else 0.7))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x.append(data[border1:border2])
        self.data_y.append(data[border1:border2])
        self.data_lengths.append(len(data[border1:border2]))

        #if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp.append(data_stamp)

    def __getitem__(self, index):

        

        #it = 0
        #ind = index
        #cumsum = dlengths[it]
        
        #while cumsum <= ind:
        #    it+=1
        #    cumsum += dlengths[it]
            
        # example dlengths: [6, 4, 6, 3, 3, 5]    , ind: 10
        
        # Calculate the cumulative sum of lengths

        if self.set_type == 0: 
            index = self.sample_arr[index] #np.random.randint(self.total_len, size=(1,))[0]
    
        it = np.searchsorted(self.cumsum, index, side='right')
    
        s_begin = self.dlengths[it] - (self.cumsum[it] - index)

        #s_begin =  self.dlengths[it] - (self.cumsum[it]-index) #index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[it][s_begin:s_end]
        seq_y = self.data_y[it][r_begin:r_end]
        seq_x_mark = self.data_stamp[it][s_begin:s_end]
        seq_y_mark = self.data_stamp[it][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):

        if self.set_type == 0: 
            return int(self.total_len*0.01)

        else:
            return np.sum(np.array(self.data_lengths) - self.seq_len - self.pred_len + 1)

        #return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
