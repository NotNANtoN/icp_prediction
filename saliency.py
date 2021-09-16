import sys

from captum.attr import NoiseTunnel, Saliency, IntegratedGradients
from tqdm import tqdm
import numpy as np
import torch

sys.path.append("../artemis")

import utils
#from src.models import LitBlockLSTM, LitLSTM


class LSTMLastStep(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        # take last timestep
        out = out[:, -1, :]
        return out


class LSTMSelectFeature(torch.nn.Module):
    def __init__(self, net, idx):
        super().__init__()
        self.net = net
        self.idx = idx
    
    def forward(self, x):
        out = self.net(x)
        if self.idx is not None:
            out = out[..., self.idx]
            out = out.unsqueeze(-1)
        return out


class LSTMMeanStep(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    
    def forward(self, x):
        out = self.net(x)
        # take mean over all timesteps
        out = out.mean(dim=1)
        out = out.unsqueeze(1)
        return out



def get_sal_model(model, idx=None, ig=False, noise_tunnel=True):
    wrapped_model = LSTMSelectFeature(LSTMMeanStep(model), idx)
    #else:
    #    wrapped_model = LSTMSelectFeature(model, idx)
    if ig:
        sal_model = IntegratedGradients(wrapped_model)
    else:
        sal_model = Saliency(wrapped_model)
    if noise_tunnel:
        sal_model = NoiseTunnel(sal_model)
    return sal_model


def get_attrs(model, absolute=False, block_size=24, idx=None, int_parts=True):
    # init
    if idx is None:
        idx = 0
    model.batch_size = 1
    ds = model.validset
    epochs = 1
    model.cuda()
    sal_model = saliency.get_sal_model(model, idx)
    # collect
    attrs = []
    for epoch in tqdm(range(epochs), disable=epochs == 1):
        for data, target, idcs, lens in tqdm(ds, disable=epochs != 1): 
            # Cut padded values out:
            data = data[:lens[0]].unsqueeze(0)
            # To GPU:
            data = data.cuda()
            # Add requires grad for saliency:
            data.requires_grad = True
            if len(data[0]) < block_size:
                continue
            split_target = torch.split(target, block_size, dim=1)
            for split_idx, _pat_split in enumerate(torch.split(data, block_size, dim=1)):
                if split_idx == (len(data[0]) // block_size) - 1 and len(data[0]) % block_size != 0:
                    break
                #print(split_target[split_idx][0, -1, idx])
                if int_parts and split_target[split_idx][0, -1, idx] == 0:
                    #print("SKIP")
                    continue
                # Calculate Atrribution:
                out = model(_pat_split)
                attribution = sal_model.attribute(_pat_split, nt_type="smoothgrad", nt_samples=50,
                                                     stdevs=0.01, abs=absolute, target=0).cpu()
                attrs.append(attribution.squeeze(0))
    #attrs = pad_and_mask(attrs)
    attrs = torch.stack(attrs).numpy()
    return attrs


def calc_median(ds, per_step=False):
    pats = []
    for pat_id in range(len(ds)):
            pat_data, _, _, len_ = ds[pat_id]
            pat_data = pat_data[:len_]
            pats.append(pat_data)    
    padded_pats = utils.pad_and_mask(pats, attr=True)

    median_per_step = np.ma.median(padded_pats, 0)

    if per_step:
        median = median_per_step
    else:
        median = np.expand_dims(np.ma.median(median_per_step, 0), 0)

    median = torch.from_numpy(median).float()
    return median


def get_baseline(ds, ig, pat_id=None, median_baseline=False, median_step_baseline=False):
    if not ig:
        return None
    if median_baseline or median_step_baseline:
        baseline_data = calc_median(ds, per_step=median_step_baseline)
    else:
        id_list_no_pat_id = list(range(len(ds)))
        id_list_no_pat_id.remove(pat_id)
        baseline_id = np.random.choice(id_list_no_pat_id)
        
        baseline_data, _, idx, len_ = ds[baseline_id]
        baseline_data = baseline_data[:len_]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return baseline_data.float().unsqueeze(0).to(device)
        

def get_all_attrs(model, absolute=False, target_idx=None, desired_target_val=None, perc=None, agg=True, ds=None, ig=False, median_baseline=False, median_step_baseline=False, num_baselines=1, sg_std=0.01, zero_baseline=False, noise_tunnel=True):
    if ds is None:
        ds = model.val_dataloader().dataset
    ids = range(len(ds.inputs))
    if perc is not None and perc != 1.0:
        ids = ids[:int(len(ids) * perc)]
        print("Only processing ", len(ids), " ids")
    if zero_baseline:
        baseline_data = None
    elif median_baseline or median_step_baseline:
        baseline_data = get_baseline(ds, ig, median_baseline=median_baseline, median_step_baseline=median_step_baseline)

    attrs = []
    for pat_id in tqdm(ids):
        pat_attrs = []
        for _ in range(num_baselines):
            if not (median_baseline or median_step_baseline or zero_baseline):
                baseline_data = get_baseline(ds, ig, pat_id=pat_id)
            
            attr = get_attr_single(model, pat_id, absolute=absolute, target_idx=target_idx, ds=ds, ig=ig, 
                                   baseline_data=baseline_data, sg_std=sg_std,noise_tunnel=noise_tunnel)
            pat_attrs.append(attr)
        
        attr = np.mean(pat_attrs, axis=0)
        
        if len(attr) > 0:
            attrs.append(attr)
        #print("Attr shape: ", attr.shape)
    if agg:
        attrs = utils.pad_and_mask(attrs, attr=True)
        
        print("All attrs shape: ", attrs.shape)
    return attrs


def get_attr_single(model, pat_id, absolute=False, target_idx=None, ds=None, ig=False, baseline_data=None, sg_std=0.01, noise_tunnel=True):
    if ds is None:
        ds = model.val_dataloader().dataset
    pat_data, pat_target = ds[pat_id]
    pat_len = len(pat_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pat_data = pat_data.unsqueeze(0).to(device)
    if target_idx is None:
        target_idx = 0
    model = get_sal_model(model, target_idx, noise_tunnel=noise_tunnel, ig=ig)
    
    pat_data.requires_grad=True
    # check if we need to cut or pad the baseline data if we have some
    if ig and baseline_data is not None:
        pat_len = pat_data.shape[1]
        if baseline_data.shape[1] > pat_len:
            # cut to pat_data len
            baseline_data = baseline_data[:, :pat_len, :]
        elif baseline_data.shape[1] < pat_len:
            # pad by repeating last timestep of baseline
            pad_by = pat_data.shape[1] - baseline_data.shape[1]
            num_feats = pat_data.shape[-1]
            last_bl_step = baseline_data[:, -1, :].unsqueeze(1)
            pad_tensor = torch.ones(1, pad_by, num_feats, device=pat_data.device) * last_bl_step
            baseline_data = torch.cat([baseline_data, pad_tensor], dim=1)                                
    # set attribution call kwargs
    attr_kwargs = {}
    if noise_tunnel:
        attr_kwargs["nt_samples"] = 50
        attr_kwargs["nt_type"] = "smoothgrad"
        attr_kwargs["stdevs"] = sg_std
    if ig:
        attr_kwargs["n_steps"] = 50
        attr_kwargs["baselines"] = baseline_data
        attr_kwargs["internal_batch_size"] = 256
    else:
        attr_kwargs["abs"] = absolute
    # calc attribution
    attrs = model.attribute(pat_data, **attr_kwargs)
    attrs = attrs.detach().cpu().squeeze(0).numpy()
    return attrs


def get_sal_list(model, idx, perc, agg, ds=None, ig=False, **sal_kwargs):
    model.train()
    attrs = get_all_attrs(model, target_idx=idx, perc=perc, agg=False, ds=ds, ig=ig, **sal_kwargs)
    summed_per_pat = [attr for attr in attrs]
    return summed_per_pat
