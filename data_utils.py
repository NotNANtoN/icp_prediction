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
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=16, pin_memory=False, collate_fn=seq_pad_collate, persistent_workers=1)            
    return dl


def to_torch(seq_list):
    return [torch.from_numpy(seq.to_numpy()).clone().float() for seq in seq_list]


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
    def __init__(self, df, db_name,
                 batch_size: int = 32, random_starts=False, 
                 min_len=10, train_noise_std=0.0, 
                 fill_type="pat_ema", flat_block_size=0, 
                 target_name = "ICP_Vital", target_nan_quantile=0,
                 block_size=0,
                 max_len=0,
                ):
        super().__init__()
        
        # dataset choice
        self.target_name = target_name
        # hyperparams
        self.random_starts = random_starts
        self.train_noise_std = train_noise_std
        self.min_len = min_len
        self.batch_size = batch_size
        self.fill_type = fill_type
        self.flat_block_size = flat_block_size
        self.target_nan_quantile = target_nan_quantile
        self.block_size = block_size
        self.max_len = max_len
        
        # construct targets and drop respective columns
        diagnose_cols = [col for col in df.columns if "diagnose" in col.lower()]
        if target_name == "ICP_Vital":
            self.regression = True
            self.classification = False
            # remove CPP as it is dependent on ICP and we want to predict ICP
            icp_cols = [col for col in df.columns if "ICP" in col]
            cpp_cols = [col for col in df.columns if "CPP" in col]
            drop_cols = icp_cols + cpp_cols + diagnose_cols 
            df["target"] = df[target_name]
            df = df.drop(columns=drop_cols)
        elif target_name.startswith("long_icp_hypertension"):
            self.regression = False
            self.classification = True
            hyper_thresh = 22.0 # value above which ICP val is considered critical
            hours_thresh = 2 # number of hours to be above to be considered a long phase
            hours_forecasted = int(target_name.split("_")[-1])  # e.g. "long_icp_hypertension_2"

            def add_hypertension_target(seq, hyper_thresh, hours_thresh, hours_forecasted):
                seq = seq.sort_values("rel_time", ascending=True)
                targets = []
                tension_count = 0
                for icp_val in seq["ICP_Vital_max"].iloc[hours_forecasted:]:
                    if icp_val > hyper_thresh:
                        tension_count += 1
                    else:
                        tension_count = 0
                    target = float(tension_count > hours_thresh)
                    targets.append(target)
                targets.extend([np.nan] * min(hours_forecasted, len(seq)))
                return pd.Series(targets)

            targets = df.groupby("Pat_ID").apply(lambda pat: 
                                                 add_hypertension_target(pat, hyper_thresh, hours_thresh, hours_forecasted))
            # assign to column
            df["target"] = list(targets)
            # drop columns
            drop_cols = diagnose_cols 
            df = df.drop(columns=drop_cols)
        
        
        # apply splits
        self.df = df
        train_data = df[df["split"] == "train"].drop(columns=["split"])
        val_data = df[df["split"] == "val"].drop(columns=["split"])
        test_data = df[df["split"] == "test"].drop(columns=["split"])
        
        non_input_cols = ["Pat_ID", "target"]
        input_cols = [col for col in train_data.columns if col not in non_input_cols]
        
        # clamp extremas according to pre-calculated vals and store them
        # calc quantiles according to train data
        self.upper_quants = torch.tensor(train_data[input_cols].quantile(0.9999))
        self.lower_quants = torch.tensor(train_data[input_cols].quantile(1 - 0.9999))

        # calc mean, median, std
        self.means, self.medians, self.stds = train_data[input_cols].mean(), train_data[input_cols].median(), train_data[input_cols].std()
        self.means, self.medians, self.stds = (torch.tensor(x).float() for x in (self.means, self.medians, self.stds))
        self.mean_train_target = train_data["target"].mean()
        self.mean_train_target_pat = train_data.groupby("Pat_ID").apply(lambda pat: pat["target"].mean()).mean()
        
        # create datasets
        self.train_ds = SequenceDataset(train_data, self.target_name, train=True,
                                        random_starts=self.random_starts,
                                        block_size=self.block_size,
                                        min_len=self.min_len, 
                                        max_len=self.max_len,
                                        train_noise_std=self.train_noise_std,
                                        flat_block_size=self.flat_block_size,
                                        target_nan_quantile=self.target_nan_quantile) if train_data is not None else None
        self.val_ds = SequenceDataset(val_data, self.target_name, train=False, random_starts=False, block_size=self.block_size, 
                         train_noise_std=0.0, flat_block_size=self.flat_block_size,
                                     max_len=self.max_len,) if val_data is not None else None
        self.test_ds = SequenceDataset(test_data, self.target_name, train=False, random_starts=False, block_size=self.block_size, 
                         train_noise_std=0.0, flat_block_size=self.flat_block_size,
                                      max_len=self.max_len,) if test_data is not None else None
        self.datasets = [ds for ds in [self.train_ds, self.val_ds, self.test_ds] if ds is not None]
        
        self.feature_names = self.train_ds.feature_names
        self.setup_completed = False
         
            
        #self.initial_setup()
        
    def setup(self, stage: Optional[str] = None): 
        # this method should only be called shortly before training
        
        if not self.setup_completed:
            # (yeo not implemented as on UMAP it just looks worse than not using it)
            # potentially get yeo-john lambdas from train dataset
            # potentially apply lambdas to all datasets
            for ds in self.datasets:
                ds.preprocess_all(fill_type=self.fill_type, median=self.medians, mean=self.means, std=self.stds, lower_quants=self.lower_quants, upper_quants=self.upper_quants)
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

    def val_dataloader(self):
        return create_dl(self.val_ds, bs=self.batch_size) if self.val_ds is not None else None

    def test_dataloader(self):
        return create_dl(self.test_ds, bs=self.batch_size) if self.test_ds is not None else None
    
    def preprocess(self, pat):
        return self.train_ds.preprocess(pat, self.fill_type, median=self.medians, mean=self.means, std=self.stds, lower_quants=self.lower_quants, upper_quants=self.upper_quants)

        
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, target_name, train, random_starts=1, block_size=0, min_len=10, train_noise_std=0.1, dbs="all", flat_block_size=0, target_nan_quantile=0, max_len=0):
        self.random_starts = random_starts
        self.min_len = min_len
        self.max_len = max_len
        self.block_size = block_size
        self.train_noise_std = train_noise_std
        self.train = train
        self.flat_block_size = flat_block_size
        self.target_nan_quantile = target_nan_quantile
        
        if target_nan_quantile > 0:
            data = data.copy()
            low_quant = data["target"].quantile(1 - target_nan_quantile)
            high_quant = data["target"].quantile(target_nan_quantile)
            data.loc[data["target"] > high_quant, "target"] = np.nan
            data.loc[data["target"] < low_quant, "target"] = np.nan
        
        # copy each sequence to not modify it outside of here
        data = [data[data["Pat_ID"] == pat_id].drop(columns=["Pat_ID"]) for pat_id in data["Pat_ID"].unique()]
        self.raw_inputs = to_torch([p.drop(columns=["target"]) for p in data])
        self.targets = to_torch([p["target"] for p in data])
        
        self.feature_names = list(data[0].drop(columns=["target"]).columns)
        
        lens = [len(pat) for pat in self.raw_inputs]
        self.ids = np.concatenate([[i] * lens[i] for i in range(len(lens))])
        self.steps = np.concatenate([np.arange(lens[i]) for i in range(len(lens))])
        
        self.all_preprocessed = False

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        seq_len = len(x)
        
        
        if self.block_size:
            total_len = self.block_size
            if self.train:
                start_idx = random.randint(0, seq_len - total_len) if seq_len > total_len else 0
            else:
                start_idx = 0
            end_idx = start_idx + total_len
            x = x[start_idx: end_idx]
            y = y[start_idx: end_idx]
            if len(x) < self.block_size:
                x_padding = torch.zeros(size=(self.block_size - len(x), x.shape[1]), dtype=x.dtype)
                x = torch.cat((x, x_padding))
                y_padding = torch.zeros(size=(self.block_size - len(y),), dtype=y.dtype) * np.nan
                y = torch.cat((y, y_padding))
            #y = y[-1]
            #y = y.unsqueeze(0)
            seq_len = end_idx - start_idx
        elif self.random_starts:
            start_idx = random.randint(0, max(seq_len - self.min_len, 0))
            if start_idx > 0:
                x = x[start_idx:]
                y = y[start_idx:]
        else:
            start_idx = 0
            
        if self.max_len > 0:
            x = x[: self.max_len]
            y = y[: self.max_len]
        
        # augment data
        if self.train_noise_std:
            x = torch.normal(x, self.train_noise_std)
            
        return x.float(), y.float()
    
    def clip_pat(self, pat, lower_quants, upper_quants):
        return pat.clip(lower_quants, upper_quants)
    
    def clip(self, lower_quants, upper_quants):
        self.inputs = [self.clip_pat(pat, lower_quants, upper_quants) for pat in self.raw_inputs]

    def preprocess_all(self, fill_type, median=None, mean=None, std=None, upper_quants=None, lower_quants=None):
        if self.all_preprocessed:
            return
        
        # apply clipping
        self.clip(lower_quants, upper_quants)

        # 1. apply filling 
        self.fill(median, fill_type)

        # 2. apply normalization 
        self.norm(mean, std)

        # 3. make flat inputs for classical ML models after having preprocessed the data
        self.make_flat_arrays()
        
        self.all_preprocessed = True
        self.mean = mean
        self.median = median
        self.std = std
        
    def preprocess(self, pat, fill_type, median=None, mean=None, std=None,
                   lower_quants=None, upper_quants=None):
        pat = self.clip_pat(pat, lower_quants, upper_quants)
        pat = self.fill_pat(pat, median, fill_type)
        pat = self.norm_pat(pat, mean, std)
        return pat
    
    def get_input_median(self):
        median = self.get_agg_statistic(self.raw_inputs, lambda pat: np.ma.median(pat, axis=0), agg="median")
        return torch.from_numpy(median)
    
    def fill(self, 
             median: torch.Tensor,
             fill_type: str="pat_mean", # pat_mean, mean, pat_ema, pat_ema_mask, none
            ):
        """Fill the NaN by using the mean of the values so far or just the overall train data mean for a single patient"""
        self.inputs = [self.fill_pat(pat, median, fill_type) for pat in self.inputs]
        
    def fill_pat(self, pat, median, fill_type):
        if fill_type == "none":
            return pat
        
        pat = pat.numpy().astype(np.float32)
        median = median.numpy().astype(np.float32)
        
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
        
    def norm_pat(self, pat, mean, std):
        return (pat - mean) / std
        
    def make_flat_arrays(self, flat_block_size=None):
        if flat_block_size is None:
            flat_block_size = self.flat_block_size
        self.flat_inputs = make_flat_inputs(self.inputs, self.flat_block_size)
        self.flat_targets = np.concatenate([data for data in self.targets], axis=0)
