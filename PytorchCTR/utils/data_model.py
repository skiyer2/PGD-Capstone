import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle

# ========== Collate Function (must be global for Windows multiprocessing) ==========
def session_collate_fn(batch):
    seqs, lbls, seq_lens = zip(*batch)
    seqs = torch.stack(seqs)  # (batch, seq_len, num_features)
    lbls = torch.stack(lbls)  # (batch, seq_len)
    seq_lens = torch.tensor(seq_lens)
    return seqs, lbls, seq_lens

# ========== Dataset for Session GRU ==========
class SessionDataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.max_session_len = args.model_params.get("max_session_len", 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        session = self.data[index]
        session_len = len(session)
        session = session[-self.max_session_len:]

        seq_len = len(session)
        features = torch.tensor([s[0] for s in session], dtype=torch.float32)
        labels = torch.tensor([s[1] for s in session], dtype=torch.float32).unsqueeze(1)

        # pad if shorter than max_session_len
        if seq_len < self.max_session_len:
            pad_len = self.max_session_len - seq_len
            pad_feat = torch.zeros(pad_len, features.shape[1])
            pad_label = torch.zeros(pad_len, 1)
            features = torch.cat([pad_feat, features], dim=0)
            labels = torch.cat([pad_label, labels], dim=0)

        return features, labels, seq_len

# ========== Read Data ==========
def read_data(args):
    dataset_path = args.data_params['dataset_path']
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    if args.model_params.get("use_gru", False):
        print("Using GRU Session-based model")
        train_data = data['train_session']
        valid_data = data['valid_session']
        test_data = data['test_session']

        train_dataset = SessionDataset(train_data, args)
        valid_dataset = SessionDataset(valid_data, args)
        test_dataset = SessionDataset(test_data, args)

        collate_fn = session_collate_fn

        train_loader = DataLoader(train_dataset, batch_size=args.data_params['batch_size'],
                                  shuffle=True, num_workers=args.model_params.get("num_workers", 0),
                                  collate_fn=collate_fn)

        valid_loader = DataLoader(valid_dataset, batch_size=args.data_params['batch_size'],
                                  shuffle=False, num_workers=args.model_params.get("num_workers", 0),
                                  collate_fn=collate_fn)

        test_loader = DataLoader(test_dataset, batch_size=args.data_params['batch_size'],
                                 shuffle=False, num_workers=args.model_params.get("num_workers", 0),
                                 collate_fn=collate_fn)

        # Return dummy features for GRU (not used)
        return train_loader, valid_loader, test_loader, [], []

    else:
        print("Using non-session model")
        train_data = data['train']
        valid_data = data['valid']
        test_data = data['test']

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data[0]), torch.tensor(train_data[1]))
        valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_data[0]), torch.tensor(valid_data[1]))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data[0]), torch.tensor(test_data[1]))

        train_loader = DataLoader(train_dataset, batch_size=args.data_params['batch_size'],
                                  shuffle=True, num_workers=args.model_params.get("num_workers", 0))

        valid_loader = DataLoader(valid_dataset, batch_size=args.data_params['batch_size'],
                                  shuffle=False, num_workers=args.model_params.get("num_workers", 0))

        test_loader = DataLoader(test_dataset, batch_size=args.data_params['batch_size'],
                                 shuffle=False, num_workers=args.model_params.get("num_workers", 0))

        fix_SparseFeat = data.get('fix_SparseFeat', [])
        fix_DenseFeat = data.get('fix_DenseFeat', [])
        return train_loader, valid_loader, test_loader, fix_SparseFeat, fix_DenseFeat