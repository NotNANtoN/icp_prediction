import math
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LitSeqModel(pl.LightningModule):
    def __init__(self, feature_names, fill_type="pat_mean"):
        super().__init__()
        print("Model fill type: ", fill_type)
        self.fill_type = fill_type
        
        # save idcs of static idcs to have separate streams in model
        static_names = ['Geschlecht', 'Alter', 'Größe', 'Gewicht']
        static_idcs = [i for i, name in enumerate(feature_names)
                       if name in static_names
                       or name.startswith("Diagnose")
                       or name.startswith("DB_")]
        non_static_idcs = [i for i in range(len(feature_names)) if i not in static_idcs]
        self.register_buffer("static_idcs", torch.tensor(static_idcs))
        self.register_buffer("recurrent_idcs", torch.tensor(non_static_idcs))
        self.num_recurrent_inputs = len(self.recurrent_idcs)
        self.num_static_inputs = len(self.static_idcs)

    # PT-lightning methods
        
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.prepare_normalization()
        self.prep_dataloaders()
        
    def on_fit_start(self, *args, **kwargs):
        super().on_fit_start(*args, **kwargs)
        self.prepare_normalization()
        self.prep_dataloaders()
                
    def training_step(self, batch, batch_idx):
        hiddens = None
        inputs, targets, lens = batch
        loss = self.calc_loss(inputs, targets, hiddens, lens=lens)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        #return {"loss": loss}#, "hiddens": self.hiddens}
    
    def validation_step(self, batch, batch_idx):
        inputs, targets, lens = batch
        loss = self.calc_loss(inputs, targets, lens=lens)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
        return [optimizer], [scheduler]
    
    # custom methods
    
    def calc_loss(self, inputs, targets, hiddens=None, lens=None):
        # pred
        if hiddens is None:
            preds = self(inputs, lens=lens)
        else:
            preds = self(inputs, hiddens=hiddens, lens=lens)
        # calc loss
        mask = torch.isnan(targets).to(inputs.device)
        loss = self.loss_func(preds, targets, mask)
        return loss
    
    def prep_dataloaders(self):
        self.prep_dataloader(self.train_dataloader.dataloader)
        if self.val_dataloader() is not None:
            self.prep_dataloader(self.val_dataloader.dataloader)
        if self.test_dataloader() is not None:
            self.prep_dataloader(self.test_dataloader())
            
    def prep_dataloader(self, dataloader):
        # preprocess inputs and set them in dataset
        dataloader.dataset.inputs = [self.preprocess(pat) for pat in dataloader.dataset.inputs]
    
    def preprocess(self, pats: torch.Tensor):
        """Inputs
            pats: A patient sequence of shape [B, T, N] - B=batch, T=steps, N=features. Unnormalized, contains NaNs
        """
        normed_pats = self.normalize(pats)
        filled_pats = self.fill(normed_pats, fill_type=self.fill_type)
        return filled_pats
    
    def normalize(self, pat: torch.Tensor):
        return (pat - self.mean) / self.std
    
    def prepare_normalization(self):
        """Prepare mean and std for normalization using the train_loader
        """
        all_inputs = []
        for input_ in self.train_dataloader.dataloader.dataset.inputs:
            input_np = input_.numpy()
            input_np = np.ma.masked_array(input_np, mask=np.isnan(input_np))
            mean_pat = input_np.mean(axis=0)
            all_inputs.append(mean_pat)
        all_inputs = np.ma.masked_array(np.stack(all_inputs))
        mean = all_inputs.mean(axis=0, keepdims=True)
        std = all_inputs.std(axis=0, keepdims=True)
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
    
    def fill(self, pat: torch.Tensor, 
             fill_type="pat_mean", # pat_mean, mean, pat_ema, pat_ema_mask
            ):
        """Fill the NaN by using the mean of the values so far or just the overall train data mean for a single patient"""
        nan_mask = torch.isnan(pat)
        if fill_type == "pat_mean":
            count = (~nan_mask).cumsum(dim=0)
            # calc cumsum without nans
            zero_filled = pat.clone()
            zero_filled[nan_mask] = 0
            cumsum = zero_filled.cumsum(dim=0)
            # calc mean until step:
            mean_until_step = cumsum / count
            # fill mean until step
            #print(pat.shape, nan_mask.shape, self.mean.shape, self.mean.repeat(pat.shape).shape, self.mean.repeat(pat.shape[0], 1))
            pat[nan_mask] = mean_until_step[nan_mask]            
        elif fill_type == "pat_ema":        
            ema_val = 0.9
            # init ema
            ema = pat[0]
            ema[torch.isnan(ema)] = 0
            # run ema
            ema_steps = []
            for pat_step in pat:
                pat_step[torch.isnan(pat_step)] = 0
                ema = ema_val * ema + (1 - ema_val) * pat_step
                ema_steps.append(ema.clone())
            pat = torch.stack(ema_steps)
        elif fill_type == "pat_ema_mask":
            ema_val = 0.3
            # init ema
            ema = pat[0]
            ema[torch.isnan(ema)] = 0
            # run ema
            ema_steps = []
            for pat_step in pat:
                mask = torch.isnan(pat_step)
                ema[~mask] = ema_val * ema[~mask] + (1 - ema_val) * pat_step[~mask]
                ema_steps.append(ema.clone())
            pat = torch.stack(ema_steps)

        # always fill remaining NaNs with the mean
        nan_mask = torch.isnan(pat)
        pat[nan_mask] = self.mean.repeat(pat.shape[0], 1)[nan_mask]
        
        assert torch.isnan(pat).sum() == 0, "NaNs still in tensor after filling!"
        
        return pat
            

class LitRNN(LitSeqModel):
    def __init__(self, feature_names, hidden_size=256, dropout_val=0.2, lstm_layers=1, lr=0.001, **kwargs):
        super().__init__(feature_names, **kwargs)
                

        self.hidden_size = hidden_size
        self.out_size = 1
        self.lr = lr
        
        # define model
        # recurrent part
        self.rnn = torch.nn.LSTM(self.num_recurrent_inputs,
                           self.hidden_size,
                           num_layers=lstm_layers,
                           batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=[self.hidden_size])
        # static part
        self.encode_static = torch.nn.Sequential(
            torch.nn.Linear(self.num_static_inputs, self.hidden_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(True),
            torch.nn.LayerNorm(normalized_shape=[self.hidden_size]),
        )
        # out part
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size), 
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.out_size))
        self.hiddens = (torch.zeros(0), torch.zeros(0))  # init for TorchScript
        # define loss 
        self.loss_func = SequentialLoss()
        
    def forward(self, 
                x: torch.Tensor, 
                lens: Optional[torch.Tensor] = None, 
                hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        recurrent_stream = x[..., self.recurrent_idcs]
        static_stream = x[..., self.static_idcs]
        
        # encode recurrence using rnn
        if lens is not None:
            recurrent_stream = pack_padded_sequence(recurrent_stream, lens.cpu(), batch_first=True, enforce_sorted=False)
        recurrent_stream, self.hiddens = self.rnn(recurrent_stream)
        if lens is not None:
            recurrent_stream, lens = pad_packed_sequence(recurrent_stream, batch_first=True)
        recurrent_stream = self.layer_norm(recurrent_stream)
        
        # encode static inputs
        static_stream = self.encode_static(static_stream)
            
        # apply mlp to transform        
        x = self.out_mlp(static_stream + recurrent_stream)
        return x

    
class LitTransformer(LitSeqModel):
    def __init__(self, 
                 ntoken,  # vocab_size or num features
                 ninp=256, # embedding dimension
                 nhead=2, # num attention heads
                 nhid=256, #the dimension of the feedforward network model in nn.TransformerEncoder
                 nlayers=2, # the number of heads in the multiheadattention models
                 dropout=0.2,
                 lr=0.001,
                 **kwargs):
        super().__init__(**kwargs)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, 
                                                          nhead, 
                                                          dim_feedforward=nhid, 
                                                          dropout=dropout, 
                                                          activation="gelu", 
                                                          batch_first=True)
        # (d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, device=None, dtype=None)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Linear(ntoken, ninp)  #Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = torch.nn.Sequential(torch.nn.Linear(ninp, ninp), 
                                           torch.nn.ReLU(True), 
                                           torch.nn.Linear(ninp, 1)
                                          )
        
        self.lr = lr
                
        # define loss func
        self.loss_func = SequentialLoss()
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, hiddens=None, lens=None):
        # bring into sequence first
        #src = src.permute(1, 0, 2)
        
        # create mask
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        # forward pass
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        
        # bring into batch first again
        #output = output.permute(1, 0, 2)
        
        return output
    
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SequentialLoss(torch.nn.Module):#jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.regression_loss_func = torch.nn.MSELoss(reduction='mean')

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
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
        return loss
