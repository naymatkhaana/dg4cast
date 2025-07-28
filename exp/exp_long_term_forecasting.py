from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_corr) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_corr = batch_corr.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_corr)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_corr)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                #print("pred.shape::",pred.shape)

                preds.append(pred)
                trues.append(true)

                loss = criterion(pred, true)

                total_loss.append(loss)

                #preds2 = np.concatenate(preds, axis=0)
                #trues2 = np.concatenate(trues, axis=0)

                #_, mse3, _, _, _, _, _ = metric(np.array(pred), np.array(true),np.array(pred), np.array(true))

                #print("shape(pred) ndividual:", pred.shape,"mse3:", mse3, "loss:", loss)


                #mae, mse, rmse, mape, mspe, smape, mae_inv = metric(np.array(preds2), np.array(trues2),np.array(preds2), np.array(trues2))
            
                #print("shape(preds2):", preds2.shape, "len(total_loss)",len(total_loss),"mse:", mse, "total_loss:", np.average(total_loss))

                #_, mse2, _, _, _, _, _ = metric(np.array(pred), np.array(true),np.array(pred), np.array(true))

                #print("shape(pred) ndividual:", pred.shape,"mse2:", mse2, "loss:", loss)
                
                #if pred.shape[0] < 32:
                #    print(":::::::: pred.shape:::::::::::::::", pred.shape, true.shape)

                #    _, mse2, _, _, _, _, _ = metric(np.array(pred), np.array(true),np.array(pred), np.array(true))
                #    print(":::::::::::::: mse2: ::::::::::", mse2)
                #    print(":::::::: non zero elements", np.count_nonzero(pred), np.count_nonzero(true), "loss: ", criterion(pred,true), mse2)

        #preds2 = np.concatenate(preds, axis=0)
        #trues2 = np.concatenate(trues, axis=0)
        #print("***************** preds2.shape, trues2.shape ::::::::::::::",preds2.shape, trues2.shape )
        #for i in range(7):
        #    _,mse,_,_,_,_,_=metric(np.array(preds2[i*32:(i+1)*32,:,:]), np.array(trues2[i*32:(i+1)*32,:,:]),np.array(preds2[i*32:(i+1)*32,:,:]), np.array(trues2[i*32:(i+1)*32,:,:]))
        #    print("******* preds2[i*32:(i+1)*32,:,:].shape",preds2[i*32:(i+1)*32,:,:].shape,"******* trues2[i*32:(i+1)*32,:,:].shape",trues2[i*32:(i+1)*32,:,:].shape, mse)


        #mae, mse, rmse, mape, mspe, smape, mae_inv = metric(np.array(preds2), np.array(trues2),np.array(preds2), np.array(trues2))

        #print("mse::::::::::::::",mse, np.sum((preds2 - trues2) ** 2)/np.count_nonzero(preds2))

        #print("total_loss::::::::::::::",total_loss)
        total_loss = np.average(total_loss)
        #print("total_loss::::::::::::::",total_loss)

        #preds = np.array(preds)
        #trues = np.array(trues)
        #preds = np.concatenate(preds, axis=0)
        #trues = np.concatenate(trues, axis=0)


        #mae, mse, rmse, mape, mspe, smape, mae_inv = metric(preds, trues,preds, trues)

        #print("mse:", mse, "total_loss:", total_loss)
        self.model.train()

        return total_loss

    def retrain(self, setting):
        nroll = self.args.num_test-self.args.pred_len
        preds_list = []; trues_list = []; preds_inverse_list = []; trues_inverse_list = [] 

        print("**************** nroll:",nroll)

        for i in range(nroll):
            print("self.args.num_train",self.args.num_train)
            print("self.args.num_test",self.args.num_test)


            self.train(setting)
            preds, trues, preds_inverse, trues_inverse = self.test(setting)
            preds_list.append(preds); trues_list.append(trues); preds_inverse_list.append(preds_inverse); trues_inverse_list.append(trues_inverse) 

            self.args.num_train+=1
            self.args.num_test-=1


        preds_list = np.concatenate(preds_list, axis=0)
        trues_list = np.concatenate(trues_list, axis=0)
        preds_inverse_list = np.concatenate(preds_inverse_list, axis=0)
        trues_inverse_list = np.concatenate(trues_inverse_list, axis=0)

        print("===============================================================================")
        print("===============================================================================")
        print("===============================================================================")


        for h in range(self.args.pred_len):    
            mae, mse, rmse, mape, mspe, smape, mae_inv = metric(preds_list[self.args.pred_len:,h,:], trues_list[self.args.pred_len:,h,:], preds_inverse_list[self.args.pred_len:,h,:], trues_inverse_list[self.args.pred_len:,h,:])
            print('horizon:{} mse:{}, mae:{}, smape:{}, dtw:{}'.format(h+1,mse, mae, smape, dtw))

        print("===============================================================================")
        hs = [6,12]
        for h in hs:    
            mae, mse, rmse, mape, mspe, smape, mae_inv = metric(preds_list[:,:h,:], trues_list[:,:h,:], preds_inverse_list[:,:h,:], trues_inverse_list[:,:h,:])
            print('average metrics: horizon upto:{} mse:{}, mae:{}, smape:{}, dtw:{}'.format(h,mse, mae, smape, dtw))


        print("===============================================================================")
        mae, mse, rmse, mape, mspe, smape, mae_inv = metric(preds_list, trues_list, preds_inverse_list, trues_inverse_list)
        print('average of horizons: mse:{}, mae:{}, smape:{}, dtw:{}'.format(mse, mae, smape, dtw))


            

    def train(self, setting, finetune_path = ''):



        self.model = self._build_model().to(self.device)


        if finetune_path != '':
            print('loading model for finetuning')
            self.model.load_state_dict(torch.load(finetune_path))

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"The number of trainable parameters is: {trainable_params}")
        #print("************************************************")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='test')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_corr) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_corr = batch_corr.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_corr)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        loss_sh = criterion(outputs[:,-self.args.pred_len+1:-self.args.pred_len+6,:], batch_y[:,-self.args.pred_len+1:-self.args.pred_len+6,:])
                        loss_sh_pre = criterion(outputs[:,-self.args.pred_len:-self.args.pred_len+1,:], batch_y[:,-self.args.pred_len:-self.args.pred_len+1,:])

                        loss = loss+loss_sh+loss_sh_pre

                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_corr)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if 'covid' in self.args.model:
                        loss = criterion(outputs[:,-self.args.pred_len+6:,:], batch_y[:,-self.args.pred_len+6:,:])

                        # Compute softmax-based weights
                        loss_detach= loss.detach()   # Prevent gradient flow through weights



                        # Weighted loss
                        loss = loss 

                    elif 'PatchTST_sep' in self.args.model:
                        loss = criterion(outputs[:,-self.args.pred_len+6:,:], batch_y[:,-self.args.pred_len+6:,:])
                        loss_sh = criterion(outputs[:,-self.args.pred_len+1:-self.args.pred_len+6,:], batch_y[:,-self.args.pred_len+1:-self.args.pred_len+6,:])
                        loss_sh_pre = criterion(outputs[:,-self.args.pred_len:-self.args.pred_len+1,:], batch_y[:,-self.args.pred_len:-self.args.pred_len+1,:])

                        # Compute softmax-based weights
                        loss_detach, loss_sh_detach, loss_sh_pre_detach = loss.detach(), loss_sh.detach() , loss_sh_pre.detach()  # Prevent gradient flow through weights

                        w = torch.exp(-loss_detach) / (torch.exp(-loss_detach) + torch.exp(-loss_sh_detach)+ torch.exp(-loss_sh_pre_detach))

                        w_sh = torch.exp(-loss_sh_detach) / (torch.exp(-loss_detach) + torch.exp(-loss_sh_detach)+ torch.exp(-loss_sh_pre_detach))

                        w_sh_pre = torch.exp(-loss_sh_pre_detach) / (torch.exp(-loss_detach) + torch.exp(-loss_sh_detach)+ torch.exp(-loss_sh_pre_detach))

                        # Weighted loss
                        loss = w * loss + w_sh * loss_sh + w_sh_pre * loss_sh_pre

                    else:
                        loss = criterion(outputs, batch_y)

                    #loss = loss+loss_sh+4*loss_sh_pre

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            #vali_loss = self.vali(vali_data, vali_loader, criterion)
            vali_loss = self.vali(test_data, test_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        preds_inverse = []
        trues_inverse = []


        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_corr) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_corr = batch_corr.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_corr)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_corr )

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if self.args.features == 'MS':
                        outputs = np.tile(outputs, [1, 1, batch_y.shape[-1]])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                #outputs = outputs[:, :, f_dim:]
                #batch_y = batch_y[:, :, f_dim:]

                pred = outputs[:, :, f_dim:]
                true = batch_y[:, :, f_dim:]

                preds.append(pred)
                trues.append(true)


                shape = batch_y.shape

                #if self.args.features == 'MS':
                #    outputs = np.tile(outputs, [1, 1, batch_y.shape[-1]])
                #    print("outputs.shape",outputs.shape)

                preds_inverse.append(test_data.inverse_transform(outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)).reshape(outputs.shape)[:, :, f_dim:])
                trues_inverse.append(test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)[:, :, f_dim:])

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if self.args.with_retrain == 1: 
                    break 

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_inverse = np.concatenate(preds_inverse, axis=0)
        trues_inverse = np.concatenate(trues_inverse, axis=0)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        preds_inverse = preds_inverse.reshape(-1, preds_inverse.shape[-2], preds_inverse.shape[-1])
        trues_inverse = trues_inverse.reshape(-1, trues_inverse.shape[-2], trues_inverse.shape[-1])
        preds_inverse = np.expm1(preds_inverse)
        trues_inverse = np.expm1(trues_inverse)


        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'
            
        smape_list = []
        for h in range(self.args.pred_len):    
            mae, mse, rmse, mape, mspe, smape, mae_inv = metric(preds[:,h,:], trues[:,h,:], preds_inverse[:,h,:], trues_inverse[:,h,:])
            print('horizon:{} mse:{}, mae:{}, smape:{}, dtw:{}'.format(h+1,mse, mae, smape, dtw))
            smape_list.append(smape)

        print("===============================================================================")
        hs = [6,12]
        for h in hs:    
            mae, mse, rmse, mape, mspe, smape, mae_inv = metric(preds[:,:h,:], trues[:,:h,:], preds_inverse[:,:h,:], trues_inverse[:,:h,:])
            print('average metrics: horizon upto:{} mse:{}, mae:{}, smape:{}, dtw:{}'.format(h,mse, mae, smape, dtw))


        print("===============================================================================")
        mae, mse, rmse, mape, mspe, smape, mae_inv = metric(preds, trues, preds_inverse, trues_inverse)
        print('average of horizons: mse:{}, mae:{}, smape:{}, dtw:{}'.format(mse, mae, smape, dtw))
        print('mean smape over horizons: ', np.mean(np.array(smape_list)))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, smape:{}, dtw:{}'.format(mse, mae,smape, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return preds, trues, preds_inverse, trues_inverse
