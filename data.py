import random

import torch
import numpy as np
#from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def create_seq_labels(seq_list, target_name="ICP_pred"):
    mean_len = np.mean([len(pat) for pat in seq_list])
    mean_target = np.mean([seq[target_name][~seq[target_name].isna()].mean() for seq in seq_list])
    print("Mean len: ", mean_len)
    print("Mean target: ", mean_target)
    labels =  [(len(seq) < mean_len).astype(int).astype(str) +
               ((seq[target_name].mean() < mean_target).astype(int).astype(str))
               for seq in seq_list]
    return labels


def make_split(seq_list, test_size=0.2):
    indices = np.arange(len(seq_list))
    labels = create_seq_labels(seq_list)
    train_data, val_data, train_idcs, val_idcs = train_test_split(seq_list, indices, test_size=test_size, stratify=labels)
    return train_data, val_data, train_idcs, val_idcs


def create_dl(data, train, target_name, random_starts=0, min_len=10, bs=32, train_noise_std=0.1, dbs="all"):
    if not train:
        random_starts = False
        train_noise_std = 0.0
    ds = SequenceDataset(data, target_name, train, random_starts=random_starts, block_size=0, 
                         min_len=min_len, train_noise_std=train_noise_std, dbs=dbs)
    # create dl. shuffle is False because we shuffle in the IterableDataset
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=False, collate_fn=seq_pad_collate)            
    return dl


def to_torch(seq_list):
    return [torch.from_numpy(seq.to_numpy()).clone() for seq in seq_list]


def seq_pad_collate(batch):
    inputs = [b[0] for b in batch]
    targets = [b[1].unsqueeze(-1) for b in batch]
    lens = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    #packed_inputs = pack_padded_sequence(padded_inputs, lens, batch_first=True, enforce_sorted=False)

    padded_targets = pad_sequence(targets, batch_first=True, padding_value=np.nan)
    return [padded_inputs, padded_targets, lens]

    

class SequenceDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, target_name, train, random_starts=1, block_size=0, min_len=10, train_noise_std=0.1, dbs="all"):
        self.random_starts = random_starts
        self.min_len = min_len
        self.block_size = block_size
        self.train_noise_std = train_noise_std
        self.train = train
        
        # drop some essential columns
        drop_cols = ["Vital_CPP", "Vital_ICP", "ICP_critical_short", "ICP_critical_long", "Outcome"]
        data = [seq.drop(columns=drop_cols) for seq in data]
        
        # determine DB, then drop the features for it
        # select rows of given databases
        all_dbs = ["DB_MIMIC", "DB_eICU", "DB_UKE"]
        db_list = all_dbs if dbs == "all" else ["DB_" + db for db in dbs]
        data = [seq for seq in data if seq[db_list].iloc[0].sum() == 1]
        # drop db features
        db_cols = [c for c in data[0].columns if "DB_" in c]
        data = [seq.drop(columns=db_cols) for seq in data]

        inputs = [seq.drop(columns=[target_name]) for seq in data]
        self.feature_names = list(inputs[0].columns)
        self.inputs = to_torch(inputs)
        self.targets = to_torch([seq[target_name] for seq in data])
        
    def __iter__(self):
        if self.train:
            # train sets
            while True:
                idx = random.randint(0, len(self.targets) - 1)
                yield self[idx]
        else:
            # validation and test sets
            for idx in range(len(self.targets)):
                yield self[idx]

    #def __len__(self):
    #    return len(self.targets)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        seq_len = len(x)
        
        if self.random_starts:
            start_idx = random.randint(0, max(seq_len - self.min_len, 0))
            if start_idx > 0:
                x = x[start_idx:]
                y = y[start_idx:]
                #x_new = x[start_idx:]
                #y_new = y[start_idx:]
                #seq_len -= start_idx
                # pad such that they fit into tensors again
                #x_pad = torch.stack([x[-1].clone() for _ in range(start_idx)])
                #x = torch.cat([x_new, x_pad])
                #y_pad = torch.stack([y[-1].clone() for _ in range(start_idx)])
                #y = torch.cat([y_new, y_pad])
        elif self.block_size:
            total_len =  self.block_size
            start_idx = random.randint(0, seq_len - total_len) if seq_len > total_len else 0
            end_idx = start_idx + total_len
            x = x[start_idx:end_idx]
            y = y[start_idx:end_idx]
            if len(x) < self.block_size:
                padding = torch.zeros(size=(self.block_size - seq_len, x.shape[1]), dtype=x.dtype)
                x = torch.cat((padding, x))
            y = y[-1]
            y = y.unsqueeze(0)
            seq_len = end_idx - start_idx
        else:
            start_idx = 0
            
        # augment data
        if self.train_noise_std:
            x = torch.normal(x, self.train_noise_std)
            
        return x.float(), y.float()
