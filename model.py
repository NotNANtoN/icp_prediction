import math
from typing import List, Optional, Tuple
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder
from x_transformers.x_transformers import AttentionLayers


class LitSeqModel(pl.LightningModule):
    def __init__(self, data_module, ema=0.0):
        super().__init__()        
        self.data_module = data_module
        self.ema = ema
        
        # get feature names
        feature_names = self.data_module.feature_names
        
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
        
        # define loss 
        self.loss_func = SequentialLoss()
        
        self._average_model = None
        
        
    def on_before_accelerator_backend_setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # for ema
        # copy the model before moving it to accelerator device.
        if self.ema > 0:
            with pl_module._prevent_trainer_and_dataloaders_deepcopy():
                self._average_model = deepcopy(pl_module)

    # PT-lightning methods
    def training_step(self, batch, batch_idx):
        hiddens = None
        inputs, targets, lens = batch
        loss = self.calc_loss(inputs, targets, hiddens, lens=lens)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
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
        return [optimizer]#, [scheduler]
    
    # custom methods
    
    def calc_loss(self, inputs, targets, hiddens=None, lens=None):
        # pred
        if hiddens is None:
            preds = self(inputs, lens=lens)
        else:
            preds = self(inputs, hiddens=hiddens, lens=lens)
        # calc loss
        loss = self.loss_func(preds, targets)
        return loss
    
    def preprocess(self, pat: torch.Tensor):
        """Inputs
            pats: A single patient sequence of shape [T, N] - T=steps, N=features. Unnormalized, contains NaNs
        """
        return self.data_module.preprocess(pat)

        
class LitMLP(LitSeqModel):
    def __init__(self, data_module, hidden_size=256, dropout_val=0.2,lr=0.001, **kwargs):
        super().__init__(data_module, **kwargs)
        self.save_hyperparameters()    
            
        self.hidden_size = hidden_size
        self.out_size = 1
        self.lr = lr
        
        # define model
        # static part
        self.encode_static = torch.nn.Sequential(
            torch.nn.Linear(self.num_static_inputs, self.hidden_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            #torch.nn.LayerNorm(normalized_shape=[self.hidden_size]),
            torch.nn.ReLU(True),
        )
        # recurrent part
        self.encode_recurrent = torch.nn.Sequential(
            torch.nn.Linear(self.num_recurrent_inputs, self.hidden_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            #torch.nn.LayerNorm(normalized_shape=[self.hidden_size]),
            torch.nn.ReLU(True),
        )
        # out part
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size), 
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.out_size))
        
        
    def forward(self, 
                x: torch.Tensor, 
                lens: Optional[torch.Tensor] = None, 
                hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        recurrent_stream = x[..., self.recurrent_idcs]
        static_stream = x[..., self.static_idcs]
        
        # encode recurrence using mlp
        recurrent_stream = self.encode_recurrent(recurrent_stream)
        
        # encode static inputs
        static_stream = self.encode_static(static_stream)
            
        # apply mlp to transform        
        x = self.out_mlp(static_stream + recurrent_stream)
        return x


class LitRNN(LitSeqModel):
    def __init__(self, data_module, hidden_size=256, dropout_val=0.2, rnn_layers=1, lr=0.001, rnn_type="lstm", **kwargs):
        super().__init__(data_module, **kwargs)
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.out_size = 1
        self.lr = lr
        
        # define model
        # recurrent part
        rnn_args = [self.num_recurrent_inputs, self.hidden_size]
        rnn_kwargs = {"num_layers": rnn_layers,
                     "batch_first": True,
                     "dropout": dropout_val if rnn_layers > 1 else 0}
        if rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(*rnn_args, **rnn_kwargs)
        elif rnn_type == "gru":
            self.rnn = torch.nn.GRU(*rnn_args, **rnn_kwargs)
        #self.layer_norm = torch.nn.LayerNorm(normalized_shape=[self.hidden_size])
        # static part
        self.encode_static = torch.nn.Sequential(
            torch.nn.Linear(self.num_static_inputs, self.hidden_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.LayerNorm(normalized_shape=[self.hidden_size]),
            torch.nn.ReLU(True),
        )
        # out part
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size), 
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.out_size))
        self.hiddens = (torch.zeros(0), torch.zeros(0))  # init for TorchScript
        
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
        #recurrent_stream = self.layer_norm(recurrent_stream)
        
        # encode static inputs
        static_stream = self.encode_static(static_stream)
            
        # apply mlp to transform        
        x = self.out_mlp(static_stream + recurrent_stream)
        return x

class LucidTransformer(LitSeqModel):
    def __init__(self, 
                 data_module,  # vocab_size or num features
                 ninp=256, # embedding dimension
                 nhead=2, # num attention heads
                 nhid=256, #the dimension of the feedforward network model in nn.TransformerEncoder
                 nlayers=2, # the number of heads in the multiheadattention models
                 dropout=0.2,
                 lr=0.001,
                 **kwargs):
        super().__init__(data_module, **kwargs)
        self.save_hyperparameters()
        
        self.out_size = 1
        self.lr = lr

        self.model = ContinuousTransformerWrapper(
            max_seq_len=1024,
            attn_layers=AttentionLayers(
                dim = nhid,
                depth = nlayers,
                heads = nhead,
                causal=True,
                rotary_pos_emb = True,
                ff_glu = True, # set to true to use for all feedforwards
                attn_num_mem_kv = 16, # 16 memory key / values
            ),
            dim_in = self.num_recurrent_inputs + self.num_static_inputs,
            dim_out = self.out_size,
            emb_dim = ninp,
            emb_dropout = dropout,
            use_pos_emb = True,
        )
        
    def forward(self, x, hiddens=None, lens=None):
        out = self.model(x)
        return out

class LitTransformer(LitSeqModel):
    def __init__(self, 
                 data_module,  # vocab_size or num features
                 ninp=256, # embedding dimension
                 nhead=2, # num attention heads
                 nhid=256, #the dimension of the feedforward network model in nn.TransformerEncoder
                 nlayers=2, # the number of heads in the multiheadattention models
                 dropout=0.2,
                 lr=0.001,
                 **kwargs):
        super().__init__(data_module, **kwargs)
        self.save_hyperparameters()
        
        self.out_size = 1
        self.hidden_size = ninp
        # recurrent stream
        ntoken = self.num_recurrent_inputs
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, 
                                                          nhead, 
                                                          dim_feedforward=nhid, 
                                                          dropout=dropout, 
                                                          activation="relu",)
                                                          #batch_first=True)
        # (d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, device=None, dtype=None)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Linear(ntoken, ninp)  #Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = torch.nn.Sequential(torch.nn.Linear(ninp, ninp), 
                                           torch.nn.ReLU(True), 
                                           torch.nn.Linear(ninp, ninp)
                                          )
        
        self.lr = lr
        
        # static part
        self.encode_static = torch.nn.Sequential(
            torch.nn.Linear(self.num_static_inputs, self.hidden_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            #torch.nn.LayerNorm(normalized_shape=[self.hidden_size]),
            torch.nn.ReLU(True),
        )
        # out part
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size), 
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.out_size))
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hiddens=None, lens=None):
        recurrent_stream = x[..., self.recurrent_idcs]
        static_stream = x[..., self.static_idcs]
        
        # encode static inputs
        static_stream = self.encode_static(static_stream)
        
        # bring into sequence first
        src = recurrent_stream.permute(1, 0, 2)
        # create mask
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        # forward pass
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        # bring into batch first again
        recurrent_stream = output.permute(1, 0, 2)
        
        # apply mlp to transform        
        x = self.out_mlp(static_stream + recurrent_stream)
        
        return x
    
    
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

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        """Calculates the loss and considers missing values in the loss given by mask indicating where targets are NaN"""
        # shape: [BS, LEN, FEATS]
        mask = ~torch.isnan(target)
        # Apply mask:
        num_feats = target.shape[-1]
        pred_per_pat = [pred_p[mask_p].reshape(-1, num_feats) for pred_p, mask_p in zip(pred, mask) if sum(mask_p) > 0]
        target_per_pat = [target_p[mask_p].reshape(-1, num_feats) for target_p, mask_p in zip(target, mask) if sum(mask_p) > 0]
        # calc raw loss per patient
        loss_per_pat = [self.regression_loss_func(p, t) for p, t in zip(pred_per_pat, target_per_pat)]
        loss = torch.stack(loss_per_pat).mean()
        
        #if torch.isinf(loss) or torch.isnan(loss):
        #    print("Found NAN or inf loss!")
        #    print(loss)
        #    raise ValueError("Found NAN or inf!")
        
        return loss
