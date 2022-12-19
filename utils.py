import torch
import numpy as np
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import sys
#sys.path.append('../artemis')
#from src.dataset import IcuDataset

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

    
def get_summary(target, pred, args):
    #  rnd_idx = torch.randint(0,len(target) - 10, (1,))
    #print('Random target', target[rnd_idx][0])
    #print('Corresponding prediction', [rnd_idx])
    print('Model type', args['model_type'])
    print('Block size', args['block_size'])
    print('Dataset', args['data'])


def get_trial_dir(mlrun_id, default_id):
    import os
    cwd = os.getcwd()
    trial_dir = cwd + '/mlruns/{}/{}/artifacts/'.format(default_id, mlrun_id)
    return trial_dir


def load_experiment_data(mlrun_id, default_id='0', summary=False, load_model=True, masked=True):
    trial_dir = get_trial_dir(mlrun_id, default_id)
    feature_names = torch.load(os.path.join(trial_dir, 'feature_names.pt'))
    target_names = torch.load(os.path.join(trial_dir, 'target_names.pt'))
    data = torch.load(os.path.join(trial_dir, 'data.pt'))
    target = torch.load(os.path.join(trial_dir, 'target.pt'))
    pred = torch.load(os.path.join(trial_dir, 'pred.pt'))
    args = torch.load(os.path.join(trial_dir, 'args.pt'))
    target_test = torch.load(os.path.join(trial_dir, 'target_test.pt'))
    pred_test = torch.load(os.path.join(trial_dir, 'pred_test.pt'))
    if masked:
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=math.nan)
        target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=math.nan)
        pred = torch.nn.utils.rnn.pad_sequence(pred, batch_first=True, padding_value=math.nan)
        target_test = torch.nn.utils.rnn.pad_sequence(target_test, batch_first=True, padding_value=math.nan)
        pred_test = torch.nn.utils.rnn.pad_sequence(pred_test, batch_first=True, padding_value=math.nan)

        data = np.ma.masked_array(data.numpy(), mask=torch.isnan(data))
        target = np.ma.masked_array(target.numpy(), mask=torch.isnan(target))
        pred = np.ma.masked_array(pred.numpy(), mask=torch.isnan(pred))
        target_test = np.ma.masked_array(target_test.numpy(), mask=torch.isnan(target_test))
        pred_test = np.ma.masked_array(pred_test.numpy(), mask=torch.isnan(pred_test))

    if load_model:
        model = torch.load(os.path.join(trial_dir, 'model.pt'))
    else:
        model = None
    get_summary(target, pred, args)

    return {
        "feature_names": feature_names,
        "target_names": target_names,
        "data": data,
        "target": target,
        "pred": pred,
        "target_test": target_test,
        "pred_test": pred_test,
        "args": args,
        "model": model
    }

def split_groups(list_of_splitting_items, outcomes, group):
    new_list = []
    for item in list_of_splitting_items:
        if group == "exitus":
            item = [element for element, outcome in zip(item, outcomes) if outcome]
        elif group == "survived":
            item = [element for element, outcome in zip(item, outcomes) if not outcome]
        new_list.append(item)
    return new_list


def get_icp_target_preds(target_names, target, pred):
    icp_idx = np.argwhere(target_names == 'ICP_pred').ravel()[0]
    icp_target = torch.stack([tar[:, icp_idx] for tar in target])
    icp_pred = torch.stack([pre[:, icp_idx] for pre in pred])
    return icp_target, icp_pred


def calc_mae(icp_pred, icp_target):
    return ((icp_pred - icp_target).abs()).mean(dim=1)


def take_mean_over_steps(timeseries_list, max_timesteps=None):
    timeseries_list = pad_sequence(timeseries_list, batch_first=True, padding_value=np.nan).numpy()
    if max_timesteps is not None:
        timeseries_list = timeseries_list[:, :max_timesteps]
    return np.nanmean(timeseries_list, axis=0), np.nanstd(timeseries_list, axis=0)


def pt_to_df(data, pat_ids, columns, lens):
    data = np.ma.filled(data, fill_value=math.nan)
    pat_dfs = []
    for patid, pat_d, _len in tqdm(zip(pat_ids, data, lens), total=len(data)):
        if lens is not None:
            pat_d = pat_d[:_len]
        #print(pat_d.shape)
        #print(len(columns))
        if pat_d.ndim == 3:
            pat_d = pat_d.squeeze(0)
        if not isinstance(pat_d, list):
            pat_d = pat_d.tolist()
     
        pat_df = pd.DataFrame(pat_d, columns=columns.tolist())
        pat_df['PatID'] = str(patid)
        pat_dfs.append(pat_df)
    return pd.concat(pat_dfs)


def filter_by_feature(data, target, pred, feature_names, feature):
    _idx = np.argwhere(feature_names == feature).ravel()[0]
    _idxs = [bool(pat_data[0, _idx]) for pat_data in data]
    return data[_idxs], target[_idxs], pred[_idxs]


def rmse_plot(pred, target, name, mlrun_id):
    err = ((pred - target)**2)
    mean_err = np.sqrt(err.mean(0))
    std_err = np.sqrt(err.std(0))
    exp_title = f'ICP {name} RMSE {mean_err.mean()} {mlrun_id}'
    fig, ax= plt.subplots(figsize=(12,8), dpi=100)
    fig.suptitle(exp_title)
    x = range(len(mean_err))
    plt.plot(x, mean_err)
    plt.fill_between(x, np.clip((mean_err - std_err).squeeze() , 0, 50), np.clip((mean_err + std_err).squeeze(), 0 , 50), alpha=.3)
    plt.savefig(exp_title + '.jpg')


def pad_and_mask(data, mask_value=math.nan, attr=False):
    if attr:
        #data = [pat.squeeze() for pat in data]
        #data = [pat.unsqueeze(0) if len(pat.shape) == 1 else pat for pat in data]

        # find max seq len
        max_len = max([pat.shape[0] for pat in data])
        # pad each seq to max seq len
        data = [np.concatenate([pat, np.zeros([ max_len - pat.shape[0], pat.shape[1]]) * math.nan], axis=0) for pat in data]
        # concat all seqs
        data = np.stack(data)
        #data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=mask_value)
        if math.isnan(mask_value):
            mask = np.isnan(data)
        else:
            mask = data == mask_value
        data = np.ma.masked_array(data, mask=mask)
    else:
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=mask_value)
        data = np.ma.masked_array(data, mask=(data == mask_value))

    return data

def load_data(database, args):
    paper_dataset = args.data.split('/')[-1]
    data_path = f'data/{database}/{paper_dataset}'
    print(data_path)
    baseset = IcuDataset(data_path, args)
    input_tensor_shape, out_size = baseset.get_model_input_output()
    dev_idcs = torch.load(os.path.join(data_path, "dev_idcs.pt"))
    devset = torch.utils.data.Subset(baseset, dev_idcs)

    val_idcs = torch.load(os.path.join(data_path, "val_idcs.pt"))

    train_idcs = [idx for idx in range(len(devset)) if idx not in val_idcs]

    test_idcs = torch.load(os.path.join(data_path, "test_idcs.pt"))
    testset = torch.utils.data.Subset(baseset, test_idcs)

    trainset = torch.utils.data.Subset(devset, train_idcs)
    validset = torch.utils.data.Subset(devset, val_idcs)

    min_id = pd.read_pickle(f"{data_path}/inputs.pkl")['Pat_ID'].unique().min()
    max_id = pd.read_pickle(f"{data_path}/inputs.pkl")['Pat_ID'].unique().max()

    all_idcs = torch.arange(min_id, max_id)
    print("Len baseset: ", len(baseset))
    print('Input shape', input_tensor_shape, 'Outsize', out_size)
    print("Devset size: ", len(devset))
    print("Trainset size: ", len(trainset))
    print("Validationset size: ", len(validset))
    print("Testset size: ", len(testset))
    print("Pat ID MIN", min_id)
    print("Pat ID MAX", max_id)

    return baseset, devset, trainset, validset, testset, val_idcs + min_id, test_idcs + min_id, all_idcs