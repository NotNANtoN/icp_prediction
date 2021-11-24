import math
import random
from typing import Optional

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pytorch_lightning
import numba


def get_seq_list(minutes, norm_targets, target_name, features=None, verbose=True):
    # read df
    df_path = f"data/{minutes}_/yeo_N/normalization_None/median/uni_clip_0.9999/multi_clip_N/df.pkl"
    if verbose:
        print("Reading df from: ", df_path)
    df = pd.read_pickle(df_path)
    if norm_targets:
        df[target_name] = df[target_name] / df[target_name].std()
    if features is not None:
        df = df[features + ["Pat_ID", target_name, "DB_UKE", "DB_eICU", "DB_MIMIC"]]
    # turn into seq list
    seq_list = [df[df["Pat_ID"] == pat_id].drop(columns=["Pat_ID"]) for pat_id in sorted(df["Pat_ID"].unique())]
    return seq_list


def do_fold(dev_data, test_data, dbs, random_starts, min_len, train_noise_std, batch_size, fill_type, flat_block_size, k_fold=0, num_splits=1):
    # do k fold
    if k_fold > 1:
        train_data_list, val_data_list, train_idcs, val_idcs = make_fold(dev_data, k=k_fold)
    else:
        # do train/val split
        train_data_list, val_data_list = [], []
        for i in range(num_splits):
            train_data, val_data, train_idcs, val_idcs = make_split(dev_data, test_size=0.2)
            train_data_list.append(train_data)
            val_data_list.append(val_data)
    # create data modules
    data_modules = [SeqDataModule(train_data, val_data, test_data, 
                                  dbs, 
                                  random_starts=random_starts, 
                                  min_len=min_len, 
                                  train_noise_std=train_noise_std, 
                                  batch_size=batch_size, 
                                  fill_type=fill_type, 
                                  flat_block_size=flat_block_size) 
                    for train_data, val_data in zip(train_data_list, val_data_list)]
    return data_modules



def create_seq_labels(seq_list, target_name="ICP_Vital"):
    median_len = np.median([len(pat) for pat in seq_list])
    median_target = np.median([seq[target_name][~seq[target_name].isna()].mean() for seq in seq_list])
    #print("Mean len: ", mean_len)
    #print("Mean target: ", mean_target)
    labels =  [(len(seq) < median_len).astype(int).astype(str) +
               ((seq[target_name][~seq[target_name].isna()].mean() < median_target).astype(int).astype(str))
               for seq in seq_list]
    return labels


def make_fold(seq_list, k=5):
    seq_list = np.array(seq_list, dtype=object)
    indices = np.arange(len(seq_list))
    labels = create_seq_labels(seq_list)
    folder = StratifiedKFold(n_splits=k, shuffle=True)

    splits = list(folder.split(seq_list, labels))
    train_idcs = [split[0] for split in splits]
    val_idcs = [split[1] for split in splits]
    train_data = [seq_list[idcs] for idcs in train_idcs]
    val_data = [seq_list[idcs] for idcs in val_idcs]
    return train_data, val_data, train_idcs, val_idcs


def make_split(seq_list, test_size=0.2):
    indices = np.arange(len(seq_list))
    labels = create_seq_labels(seq_list)
    train_data, val_data, train_idcs, val_idcs = train_test_split(seq_list, indices, test_size=test_size, stratify=labels, shuffle=True)
    return train_data, val_data, train_idcs, val_idcs


def create_dl(ds, bs=32):
    # create dl. shuffle is False because we shuffle in the IterableDataset for train data
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False, collate_fn=seq_pad_collate)            
    return dl


def to_torch(seq_list):
    return [torch.from_numpy(seq.to_numpy()).clone() for seq in seq_list]


def seq_pad_collate(batch):
    inputs = [b[0] for b in batch]
    targets = [b[1].unsqueeze(-1) for b in batch]
    lens = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    #packed_inputs = pack_padded_sequence(padded_inputs, lens, batch_first=True, enforce_sorted=False)

    padded_targets = pad_sequence(targets, batch_first=True, padding_value=math.nan)
    return [padded_inputs, padded_targets, lens]


def make_flat_inputs(inputs, flat_block_size, targets=None):
    if flat_block_size > 0:
        # make blocks
        num_feats = inputs[0].shape[-1]
        all_blocks = []
        for pat_idx, pat in enumerate(inputs):
            pat = pat.numpy()
            for time_idx in range(len(pat)):
                # skip block for training if targets are given
                if targets is not None and np.isnan(targets[pat_idx][time_idx]):
                    continue
                
                start_idx = max(time_idx + 1 - flat_block_size, 0)
                end_idx = time_idx + 1
                block = pat[start_idx: end_idx]
                size_diff = flat_block_size - block.shape[0]
                if size_diff > 0:
                    # pad start with nans
                    pad_prefix = np.zeros((size_diff, num_feats)) * np.nan
                    block = np.concatenate([pad_prefix, block], axis=0)
                # make flat
                block = block.reshape(-1)
                all_blocks.append(block)
        inputs = np.stack(all_blocks)
    else:
        # make flat
        inputs = np.concatenate(inputs, axis=0)
        if targets is not None:
            targets = np.concatenate(targets, axis=0)
            inputs = inputs[~np.isnan(targets)]
    return inputs


@numba.jit()
def ema_fill(pat: np.ndarray, ema_val: float, mean: np.ndarray):
    # init ema
    ema = np.ones_like(pat[0]) * mean
    # run ema
    ema_steps = np.ones_like(pat)
    for i, pat_step in enumerate(pat):
        pat_step[np.isnan(pat_step)] = 0
        ema = ema_val * ema + (1 - ema_val) * pat_step
        ema_steps[i] = ema.copy()
    return ema_steps


@numba.jit()
def ema_fill_mask(pat: np.ndarray, ema_val: float, mean: np.ndarray):
    # init ema
    ema = np.ones_like(pat[0]) * mean
    # run ema
    ema_steps = np.ones_like(pat)
    for i, pat_step in enumerate(pat):
        mask = np.isnan(pat_step)
        ema[~mask] = ema_val * ema[~mask] + (1 - ema_val) * pat_step[~mask]
        ema_steps[i] = ema.copy()
    return ema_steps


class SeqDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, dbs,
                 batch_size: int = 32, random_starts=False, min_len=10, train_noise_std=0.0, 
                 fill_type="pat_ema", flat_block_size=0
                ):
        super().__init__()
        # dataset choice
        self.target_name = "ICP_Vital" 
        self.dbs = dbs
        # hyperparams
        self.random_starts = random_starts
        self.train_noise_std = train_noise_std
        self.min_len = min_len
        self.batch_size = batch_size
        self.fill_type = fill_type
        self.flat_block_size = flat_block_size

        
        # create datasets
        self.train_ds = SequenceDataset(train_data, self.target_name, train=True, random_starts=self.random_starts, block_size=0, 
                         min_len=self.min_len, train_noise_std=self.train_noise_std, dbs=self.dbs, flat_block_size=self.flat_block_size) if train_data is not None else None
        self.val_ds = SequenceDataset(val_data, self.target_name, train=False, random_starts=False, block_size=0, 
                         train_noise_std=0.0, dbs=self.dbs, flat_block_size=self.flat_block_size) if val_data is not None else None
        self.test_ds = SequenceDataset(test_data, self.target_name, train=False, random_starts=False, block_size=0, 
                         train_noise_std=0.0, dbs=self.dbs, flat_block_size=self.flat_block_size) if test_data is not None else None
        self.datasets = [ds for ds in [self.train_ds, self.val_ds, self.test_ds] if ds is not None]
        
        self.feature_names = self.train_ds.feature_names
        self.setup_completed = False
        
        self.initial_setup()
        
    def initial_setup(self, stage: Optional[str] = None):    
        if not self.setup_completed:
            # (yeo not implemented as on UMAP it just looks worse than not using it)
            # potentially get yeo-john lambdas from train dataset
            # potentially apply lambdas to all datasets
            
            # first do for train_ds to get all statistics
            train_ds = self.train_ds
            self.median, self.mean, self.std = train_ds.preprocess_all(fill_type=self.fill_type)
            # then for all other datasets
            eval_datasets = [ds for ds in [self.val_ds, self.test_ds] if ds is not None]
            for ds in eval_datasets:
                ds.preprocess_all(fill_type=self.fill_type, median=self.median, mean=self.mean, std=self.std)
                
            # calc mean train target for later evaluation
            self.mean_train_target_pat = np.array([pat[~torch.isnan(pat)].numpy().mean() 
                                                     for pat in self.train_ds.targets
                                                    ]).mean()
            
            self.mean_train_target = np.concatenate([pat[~torch.isnan(pat)].numpy()
                                                     for pat in self.train_ds.targets
                                                    ]).mean()
                
            # done
            self.setup_completed = True
            
    def fill(self, median, fill_type):
        for ds in self.datasets:
            ds.fill(median, fill_type)
    
    def norm(self, mean, std):
        for ds in self.datasets:
            ds.norm(mean, std)
            
    def make_flat_arrays(self):
        for ds in self.datasets:
            ds.make_flat_arrays(self.flat_block_size)

    def train_dataloader(self):
        return create_dl(self.train_ds, bs=self.batch_size) if self.train_ds is not None else None
        #return self.train_dl

    def val_dataloader(self):
        return create_dl(self.val_ds, bs=self.batch_size * 4) if self.val_ds is not None else None
        #return self.val_dl

    def test_dataloader(self):
        return create_dl(self.test_ds, bs=self.batch_size * 4) if self.test_ds is not None else None
        #return self.test_dl

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass
    

class SequenceDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, target_name, train, random_starts=1, block_size=0, min_len=10, train_noise_std=0.1, dbs="all", flat_block_size=0):
        self.random_starts = random_starts
        self.min_len = min_len
        self.block_size = block_size
        self.train_noise_std = train_noise_std
        self.train = train
        self.flat_block_size = flat_block_size
        
        # copy each sequence to not modify it outside of here
        data = [seq.copy() for seq in data]
        
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
        self.raw_inputs = to_torch(inputs)
        self.targets = to_torch([seq[target_name] for seq in data])
        
        self.ids = np.concatenate([[i] * len(inputs[i]) for i in range(len(inputs))])
        self.steps = np.concatenate([np.arange(len(inputs[i])) for i in range(len(inputs))])
        
        self.all_preprocessed = False
        
    def __iter__(self):
        if self.train:
            # train sets
            while True:
                idx = random.randint(0, len(self.targets) - 1)
                yield self[idx]
        else:   
            start = 0
            end = len(self.targets)

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:  # single-process data loading, return the full iterator
                iter_start = start
                iter_end = end
            else:  # in a worker process
                 # split workload
                per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = start + worker_id * per_worker
                iter_end = min(iter_start + per_worker, end)
        
            # validation and test sets
            for idx in range(iter_start, iter_end):
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

    def preprocess_all(self, fill_type, median=None, mean=None, std=None):
        if self.all_preprocessed:
            return
        
        # 1. get median for train dataset for filling and store
        if median is None:
            median = self.get_input_median()

        # 2. apply filling with median from train
        self.fill(median, fill_type)

        # 3. get mean and std for train dataset
        if mean is None:
            mean, std = self.get_mean_std()

        # 4. apply standardization 
        self.norm(mean, std)

        # 5. make flat inputs for classical ML models after having preprocessed the data
        self.make_flat_arrays()
        
        self.all_preprocessed = True
        self.mean = mean
        self.median = median
        self.std = std

        return median, mean, std
    
    def get_input_median(self):
        median = self.get_agg_statistic(self.raw_inputs, lambda pat: np.ma.median(pat, axis=0), agg="median")
        return torch.from_numpy(median)
    
    def fill(self, 
             median: torch.Tensor,
             fill_type: str="pat_mean", # pat_mean, mean, pat_ema, pat_ema_mask
            ):
        """Fill the NaN by using the mean of the values so far or just the overall train data mean for a single patient"""
        self.inputs = [self.fill_pat(pat, median, fill_type) for pat in self.raw_inputs]
        
    def fill_pat(self, pat, median, fill_type):
        pat = pat.numpy()
        median = median.numpy()
        
        nan_mask = np.isnan(pat)
        if fill_type == "pat_median":
            count = (~nan_mask).cumsum(axis=0)
            # calc cumsum without nans
            median_filled = pat.copy()
            median_filled[nan_mask] = median[nan_mask]
            cumsum = median_filled.cumsum(axis=0)
            # calc mean until step:
            mean_until_step = cumsum / count
            # fill mean until step
            pat[nan_mask] = mean_until_step[nan_mask]            
        elif fill_type == "pat_ema":        
            ema_val = 0.9
            pat = ema_fill(pat, ema_val, median)
        elif fill_type == "pat_ema_mask":
            ema_val = 0.3
            pat = ema_fill_mask(pat, ema_val, median)

        pat = torch.from_numpy(pat)
        median = torch.from_numpy(median)
        # always fill remaining NaNs with the median
        nan_mask = torch.isnan(pat)
        pat[nan_mask] = median.repeat(pat.shape[0], 1)[nan_mask]
        
        assert torch.isnan(pat).sum() == 0, "NaNs still in tensor after filling!"
        
        return pat
    
    def get_mean_std(self):
        mean = self.get_agg_statistic(self.inputs, lambda pat: np.ma.mean(pat, axis=0))
        std = self.get_agg_statistic(self.inputs, lambda pat: np.ma.std(pat, axis=0))
        return torch.from_numpy(mean), torch.from_numpy(std)
    
    def get_agg_statistic(self, seq_list, func, agg="mean"):
        stats = np.array([func(np.ma.masked_array(seq.numpy(), mask=np.isnan(seq.numpy()))) for seq in seq_list])
        if agg == "mean":
            stat = np.ma.mean(np.ma.masked_array(stats, mask=np.isnan(stats)), axis=0)
        else:
            stat = np.ma.median(np.ma.masked_array(stats, mask=np.isnan(stats)), axis=0)
        return stat
    
    def norm(self, mean, std):
        self.inputs = [(pat - mean) / std for pat in self.inputs]
        
    def make_flat_arrays(self, flat_block_size=None):
        if flat_block_size is None:
            flat_block_size = self.flat_block_size
        self.flat_inputs = make_flat_inputs(self.inputs, self.flat_block_size)
        self.flat_targets = np.concatenate([data for data in self.targets], axis=0)
