import logging
import os
from scipy import sparse
import sys
import torch
import torch.nn.functional as F
from utils.save import save_checkpoint
from utils.evaluate import get_auc
import numpy as np
from scipy.sparse import csr_matrix
import time
from tqdm import tqdm
                     
class Trainer():
    def __init__(self, model, args, loss_function, train_loader, test_loader, valid_loader):
        self.model = model
        self.loss_function = loss_function
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        self.best_results = {'auc': np.zeros(1)}
        self.use_gru = args.model_params.get('use_gru', False)

    def train(self):
        print('Train Begin')
        Recmodel = self.model
        for epoch_counter in range(self.args.model_params['epochs']):
            Recmodel.train()
            aver_loss = 0.
            for batch_counter in tqdm(self.train_loader):
                if self.use_gru:
                    x_seq = batch_counter[0].long().to(self.args.device)      # (batch, seq_len, num_features)
                    y_seq = batch_counter[1].float().to(self.args.device)     # (batch, seq_len)
                    seq_len = batch_counter[2]                                # (batch,)
                    # Option 1: Use last event in each session for loss/AUC
                    y = y_seq[range(len(seq_len)), seq_len - 1]               # (batch,)
                    cri = self.loss_function.stageOne(x_seq, y)
                else:
                    x = batch_counter[0].long().to(self.args.device)
                    y = batch_counter[1].float().to(self.args.device)
                    cri = self.loss_function.stageOne(x, y)
                aver_loss += cri

            aver_loss = aver_loss / len(self.train_loader)
            print('Epoch:', epoch_counter, '|loss:%.4f' % aver_loss)
            self.args.file_logger.log("{}\t{:.6f}".format(epoch_counter, aver_loss))
            if self.args.model_params['eval_epoch'] is not None:
                if epoch_counter % self.args.model_params['eval_epoch'] == 0:
                    self.valid(epoch_counter)
        self.test(epoch_counter)
    
    def valid(self, epoch_counter):
        self.model.eval()
        with torch.no_grad():
            auc = 0
            for batch_counter in tqdm(self.valid_loader):
                if self.use_gru:
                    x_seq = batch_counter[0].long().to(self.args.device)
                    y_seq = batch_counter[1].float().to(self.args.device)
                    seq_len = batch_counter[2]
                    y = y_seq[range(len(seq_len)), seq_len - 1]
                    y_pred, _ = self.model(x_seq)
                    auc += get_auc(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy())
                else:
                    x = batch_counter[0].long().to(self.args.device)
                    y = batch_counter[1].float().to(self.args.device)
                    y_pred, _ = self.model(x)
                    auc += get_auc(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy())

            avg_auc = auc / (len(self.valid_loader))
            print("Validation AUC: {:.6f}".format(avg_auc))
            self.args.file_logger.log("Validation AUC: {:.6f}".format(avg_auc))

            if avg_auc > self.best_results['auc']:
                self.best_results['auc'] = avg_auc
                self.best_results['best_epoch'] = epoch_counter
                self.test(epoch_counter)
            print(self.best_results)
            self.args.file_logger.log(self.best_results)

    def test(self, epoch_counter):
        self.model.eval()
        with torch.no_grad():
            auc = 0
            for batch_counter in tqdm(self.test_loader):
                if self.use_gru:
                    x_seq = batch_counter[0].long().to(self.args.device)
                    y_seq = batch_counter[1].float().to(self.args.device)
                    seq_len = batch_counter[2]
                    y = y_seq[range(len(seq_len)), seq_len - 1]
                    y_pred, _ = self.model(x_seq)
                    auc += get_auc(y_pred.cpu(), y.cpu())
                else:
                    x = batch_counter[0].long().to(self.args.device)
                    y = batch_counter[1].float().to(self.args.device)
                    y_pred, _ = self.model(x)
                    auc += get_auc(y_pred.cpu(), y.cpu())

            avg_auc = auc / (len(self.test_loader))
            print("Test AUC: {:.6f}".format(avg_auc))
            self.args.file_logger.log("Test AUC: {:.6f}".format(avg_auc))