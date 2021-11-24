import os
import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import pandas as pd
import numpy as np
from tqdm import tqdm

#from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split


def to_torch(seq_list):
    return [torch.from_numpy(seq.to_numpy()) for seq in seq_list]


def seq_pad_collate(batch):
    inputs = [b[0] for b in batch]
    targets = [b[1].unsqueeze(-1) for b in batch]
    lens = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=np.nan)
    return [padded_inputs, padded_targets, lens]


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, target_name, random_starts=1, block_size=0, min_len=10, train_noise_std=0.1):
        self.random_starts = random_starts
        self.min_len = min_len
        self.block_size = block_size
        self.train_noise_std = train_noise_std
        
        # determine DB, then drop the features for it
        db_cols = [c for c in data[0].columns if "DB_" in c]
        data = [seq.drop(columns=db_cols) for seq in data]

        inputs = [seq.drop(columns=[target_name]) for seq in data]
        self.feature_names = list(inputs[0].columns)
        self.inputs = to_torch(inputs)
        self.targets = to_torch([seq[target_name] for seq in data])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        seq_len = len(x)
        
        if self.random_starts:
            start_idx = random.randint(0, max(seq_len - self.min_len, 0))
            if start_idx > 0:
                x_new = x[start_idx:]
                y_new = y[start_idx:]
                seq_len -= start_idx
                # pad such that they fit into tensors again
                x_pad = torch.stack([x[-1].clone() for _ in range(start_idx)])
                x = torch.cat([x_new, x_pad])
                y_pad = torch.stack([y[-1].clone() for _ in range(start_idx)])
                y = torch.cat([y_new, y_pad])

        if self.block_size:
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
            x = x + torch.zeros_like(x).normal_(0.0, self.train_noise_std)
            
        return x.float(), y.float()

 
class LitRNN(torch.nn.Module):
    def __init__(self, num_features, hidden_size=256, dropout_val=0.2, lstm_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = 1
        
        # define model
        self.rnn = torch.nn.LSTM(num_features,
                           self.hidden_size,
                           num_layers=lstm_layers,
                           batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=[self.hidden_size])
        self.out_mlp = torch.nn.Sequential(
            #torch.nn.Linear(self.hidden_size, self.hidden_size), 
            #torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.out_size))
        
        

    def forward(self, x, lens=None):
        if lens is not None:
            x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, self.hidden = self.rnn(x)
        if lens is not None:
            x, lens = pad_packed_sequence(x, batch_first=True)
        #x = self.layer_norm(x)
        x = self.out_mlp(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target, idcs, lens = batch

        if self.args.gauss_std:
            data = torch.normal(data, self.args.gauss_std)

        output = self(data, lens=lens)
        target = target[:, :output.shape[1]]
        loss_mask = torch.isnan(target)
        loss = self.loss_func(output, target, loss_mask)
        return {'loss': loss, 'log': {'train_loss': loss.item()}}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
        return [optimizer], [scheduler]


class SequentialLoss:
    def __init__(self):
        self.regression_loss_func = torch.nn.MSELoss(reduction='mean')

    def __call__(self, pred, target, mask):
        """Calculates the loss and considers missing values in the loss given by mask"""
        # shape: [BS, LEN, FEATS]
        # Apply mask:
        num_feats = target.shape[-1]
        mask = ~mask
        pred_per_pat = [pred_p[mask_p].reshape(-1, num_feats) for pred_p, mask_p in zip(pred, mask) if sum(mask_p) > 0]
        target_per_pat = [target_p[mask_p].reshape(-1, num_feats) for target_p, mask_p in zip(target, mask) if sum(mask_p) > 0]
        
        # calc raw loss per patient
        loss_per_pat = [self.regression_loss_func(p, t) for p, t in zip(pred_per_pat, target_per_pat)]
        
        loss = torch.stack(loss_per_pat).mean()
        
        verbose = False
        if verbose:
            print("pred per pat", pred_per_pat)
            print(" target per pat", target_per_pat)
            print("loss per pat", loss_per_pat)
            print("loss: ", loss)
            print()
        return loss
    
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


def create_dl(data, train, target_name, random_starts=0, min_len=10, bs=32, train_noise_std=0.1):
    shuffle = True
    if not train:
        random_starts = False
        shuffle = False
        train_noise_std = 0.0
    ds = SequenceDataset(data, target_name=target_name, random_starts=random_starts, block_size=0, min_len=min_len, train_noise_std=train_noise_std)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True, collate_fn=seq_pad_collate)            
    return dl

@torch.cuda.amp.autocast()
def calc_loss(inputs, targets, model):
    # to cuda
    inputs = inputs.to("cuda")
    targets = targets.to("cuda")
    # pred
    preds = model(inputs)
    # calc loss
    mask = torch.isnan(targets)
    loss = loss_func(preds, targets, mask)
    return loss

# fixed settings
target_name = "ICP_pred"   
# hyperparams
epochs = 20
lr = 0.003
bs = 32
norm_targets = False

train_noise_std = 0.1
random_starts = True
min_len = 10

hidden_size = 256

norm_method = "z" # z, None

                                  
# read df
df_path = f"data/yeo_N/normalization_{norm_method}/median/uni_clip_0.9999/multi_clip_N/df.pkl"
print("Reading df from: ", df_path)
df = pd.read_pickle(df_path)
if norm_targets:
    df[target_name] = df[target_name] / df[target_name].std()
# turn into seq list
seq_list = [df[df["Pat_ID"] == pat_id].drop(columns=["Pat_ID"]) for pat_id in sorted(df["Pat_ID"].unique())]

# do train/val split
train_data, val_data, train_idcs, val_idcs = make_split(seq_list, test_size=0.2)
print(train_idcs)
# create train dataset
train_dl = create_dl(train_data, train=True, target_name=target_name, random_starts=random_starts, min_len=min_len, bs=bs, train_noise_std=train_noise_std)
val_dl = create_dl(val_data, train=False, target_name=target_name, bs=bs)
num_features = train_dl.dataset.inputs[0].shape[-1]
feature_names = train_dl.dataset.feature_names

# test dataloaders   
inputs, targets, lens = next(iter(train_dl))
print(feature_names)
print(inputs.shape, targets.shape, lens)
inputs, targets, lens = next(iter(val_dl))
print(inputs.shape, targets.shape, lens)

# define model
model = LitRNN(num_features, hidden_size=hidden_size, dropout_val=0., lstm_layers=1)
# define loss
loss_func = SequentialLoss()

# test loss and model
mask = torch.isnan(targets)
preds = model(inputs)
loss = loss_func(preds, targets, mask)
print("test of loss func : ", loss)


# train a bit
opt = torch.optim.Adam(model.parameters(), lr=lr)
model.to("cuda")

from torch import autograd

# Creates gradient scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

for ep in range(epochs):
    for inputs, targets, lens in tqdm(train_dl):
        loss = calc_loss(inputs, targets, model)
        # update params
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
                
    # val loss
    val_losses = []
    with torch.no_grad():
        for inputs, targets, lens in val_dl:
            loss = calc_loss(inputs, targets, model)
            val_losses.append(loss)
        
    print(torch.mean(torch.stack(val_losses)).item())


# Eval model
all_targets = []
all_preds = []
all_times = []
all_ids = []
count = 0
with torch.no_grad():
    for inputs, targets, lens in val_dl:
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        mask = torch.isnan(targets)
        preds = model(inputs)
        times = torch.stack([torch.arange(inputs.shape[1]) for _ in range(inputs.shape[0])]).unsqueeze(-1)
        ids = torch.stack([torch.ones(inputs.shape[1]) * (count + i) for i in range(inputs.shape[0])]).unsqueeze(-1)
        count += inputs.shape[0]
        
        no_nan_times = times[~mask]
        no_nan_targets = targets[~mask]
        no_nan_preds = preds[~mask]
        no_nan_ids = ids[~mask]
        all_targets.append(no_nan_targets)
        all_preds.append(no_nan_preds)
        all_times.append(no_nan_times)
        all_ids.append(no_nan_ids)
        
all_targets = torch.cat(all_targets).cpu().flatten().numpy()
all_preds = torch.cat(all_preds).cpu().flatten().numpy()
all_ids = torch.cat(all_ids).cpu().flatten().numpy()
all_times = torch.cat(all_times).cpu().flatten().numpy()

all_errors = (all_targets - all_preds) ** 2

print("Mean/Std preds: ", np.mean(all_preds), np.std(all_preds))
print("Mean/Std targets: ", np.mean(all_targets), np.std(all_targets))

print("Max error: ", np.max(all_errors))
print("Min error: ", np.min(all_errors))
print("Mean error: ", np.mean(all_errors))

import matplotlib.pyplot as plt
import seaborn as sns
folder = "debug_plots"

torch.save(all_targets, os.path.join(folder, "all_targets.pt"))
torch.save(all_preds, os.path.join(folder, "all_preds.pt"))
torch.save(all_ids, os.path.join(folder, "all_ids.pt"))
torch.save(all_times, os.path.join(folder, "all_times.pt"))
torch.save(all_errors, os.path.join(folder, "all_errors.pt"))




#plt.scatter(all_targets, all_errors)
sns.scatterplot(all_targets, all_errors)
plt.savefig(os.path.join(folder, "scatter.png"))

sns.boxplot(all_errors)
plt.savefig(os.path.join(folder, "error_boxplot.png"))

sns.violinplot(all_errors)
plt.savefig(os.path.join(folder, "error_violinplot.png"))

plt.violinplot([all_preds, all_targets], showmeans=True, showmedians=False,
        showextrema=True)
plt.savefig(os.path.join(folder, "preds_targets_violinplot.png"))







