import math
from typing import List, Optional, Tuple
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder
from x_transformers.x_transformers import AttentionLayers


class LitSeqModel(pl.LightningModule):
    def __init__(self, data_module, max_epochs=5, weight_decay=0.1, use_macro_loss=True, 
                 use_pos_weight=True, use_nan_embed=False, lr=0.001, use_huber=False,
                 use_static=True, freeze_nan_embed=False, norm_nan_embed=False, nan_embed_size=512):
        super().__init__()        
        self.data_module = data_module
        self.regression = data_module.regression
        self.use_macro_loss = use_macro_loss
        self.use_nan_embed = use_nan_embed
        self.lr = lr
        self.use_huber = use_huber
        self.use_static = use_static
        self.freeze_nan_embed = freeze_nan_embed
        self.nan_embed_size = nan_embed_size
        self.norm_nan_embed = norm_nan_embed
        
        # get feature names
        feature_names = self.data_module.feature_names
        
        if use_static:
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
        else:
            self.num_recurrent_inputs = len(feature_names)
            self.num_static_inputs = 0
            self.static_idcs = None
            self.recurrent_idcs = list(range(len(feature_names)))
        self.num_inputs = len(feature_names)
        
        # define loss 
        targets = torch.cat(data_module.train_ds.targets)
        pos_frac = targets[~torch.isnan(targets)].mean()
        self.pos_weight = (1 - pos_frac) / pos_frac if use_pos_weight else None
        self.loss_func = SequentialLoss(self.regression, self.use_macro_loss, self.pos_weight, self.use_huber)
        
        if self.use_nan_embed:
            from nan_emb import NanEmbed
            emb_size = self.nan_embed_size
            self.embed = NanEmbed(self.num_inputs, emb_size)
            #self.embed = torch.jit.script(self.embed)

            if self.freeze_nan_embed:
                for p in self.embed.parameters():
                    p.requires_grad = False

            if self.norm_nan_embed:
                self.embed = torch.nn.Sequential(self.embed, torch.nn.LayerNorm(emb_size))
            
            self.num_recurrent_inputs = emb_size
            self.num_static_inputs = 0
            self.static_idcs = None
            self.recurrent_idcs = torch.tensor(list(range(emb_size)))
        
        
        self._average_model = None
        
        self.max_epochs = max_epochs
        self.steps_per_epoch = len(data_module.train_dataloader())
        self.weight_decay = weight_decay
        
    def forward(self, x, *args, **kwargs):
        if self.use_nan_embed:
            x_emb = self.embed(x.reshape(-1, x.shape[-1]))#[:, 0])
            x = x_emb.reshape(*x.shape[:-1], x_emb.shape[-1])

        preds = self.make_preds(x, *args, **kwargs)
        if not self.regression:
            preds = torch.sigmoid(preds)
        return preds
    # custom methods

    # PT-lightning methods
    def training_step(self, batch, batch_idx):
        hiddens = None
        inputs, targets, lens = batch
        loss, preds = self.calc_loss(inputs, targets, hiddens, lens=lens)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        lr = self.scheduler.get_last_lr()[0]
        self.log('lr', lr, on_epoch=False, on_step=True, prog_bar=False, logger=True)
        return loss
        #return {"loss": loss}#, "hiddens": self.hiddens}
    
    def validation_step(self, batch, batch_idx):
        inputs, targets, lens = batch
        loss, preds = self.calc_loss(inputs, targets, lens=lens)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return preds, targets
    
    def validation_epoch_end(self, val_step_output_list):
        # calculate f1_score, accuracy etc
        preds = torch.cat([step_out[0].flatten() for step_out in val_step_output_list]).cpu().squeeze().float().numpy()
        targets = torch.cat([step_out[1].flatten() for step_out in val_step_output_list]).cpu().squeeze().float().numpy()
        # remove NaNs
        mask = ~np.isnan(targets)
        preds = preds[mask]
        targets = targets[mask]
        
        if self.regression:
            try:
                r2 = sklearn.metrics.r2_score(targets, preds)
                mse = sklearn.metrics.mean_squared_error(targets, preds)
                rmse = np.sqrt(mse)
                mae = sklearn.metrics.mean_absolute_error(targets, preds)
            except ValueError:
                print("ValueError: r2_score")
                print(targets.shape, preds.shape)
                print(np.isinf(targets).sum(), np.isinf(preds).sum())
                print(np.isnan(targets).sum(), np.isnan(preds).sum())
                print(targets, preds)

                # to stop training, raise a KeyboardInterrupt
                raise KeyboardInterrupt

            
            self.log("val_r2", r2, on_epoch=True, prog_bar=True)
            self.log("val_mse", mse, on_epoch=True, prog_bar=True)
            self.log("val_rmse", rmse, on_epoch=True, prog_bar=True)
            self.log("val_mae", mae, on_epoch=True, prog_bar=True)
            
        else:
            # average precision
            ap = sklearn.metrics.average_precision_score(targets, preds, average="macro", pos_label=1)
            self.log('val_ap', ap, on_epoch=True, prog_bar=False)
            # auc - per class and macro-averaged
            #print("Shapes: ", targets.shape, preds.shape)
            auc_micro = sklearn.metrics.roc_auc_score(targets, preds, average="micro")
            #auc_macro = sklearn.metrics.roc_auc_score(targets, preds, average="macro")
            #self.log('val_auc_macro', auc_macro, on_epoch=True, prog_bar=True)
            self.log('val_auc_micro', auc_micro, on_epoch=True, prog_bar=True)

            # metrics based on binary predictions
            binary_preds = (preds > 0.5).astype(int)
            self.log("val_acc_micro", sklearn.metrics.accuracy_score(targets.reshape(-1), binary_preds.reshape(-1)), on_epoch=True)
            #macro_acc = (targets == binary_preds).astype(float).mean(axis=0).mean()
            #self.log("val_acc_macro", macro_acc, on_epoch=True)
            #self.log("val_f1_macro", sklearn.metrics.f1_score(targets, binary_preds, average="macro"), on_epoch=True, logger=True)
            self.log("val_f1_micro", sklearn.metrics.f1_score(targets, binary_preds, average="micro"), on_epoch=True, logger=True)

            # log diagnostics of output distribution
            preds_for_pos = preds[targets == 1]
            preds_for_neg = preds[targets == 0]
            pos_mean = preds_for_pos.mean()
            neg_mean = preds_for_neg.mean()
            self.log("debug/pos_preds_mean", pos_mean, on_epoch=True)
            self.log("debug/pos_preds_std", preds_for_pos.std(), on_epoch=True)
            self.log("debug/neg_preds_mean", neg_mean, on_epoch=True)
            self.log("debug/neg_preds_std", preds_for_neg.std(), on_epoch=True)
            self.log("debug/preds_mean", preds.mean(), on_epoch=True)
            self.log("debug/preds_std", preds.std(), on_epoch=True)
            self.log("debug/preds_mean_diff", pos_mean - neg_mean, on_epoch=True)
    

    def configure_optimizers_old(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
        return [optimizer]#, [scheduler]
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                              lr=self.lr,#5e-5,
                              betas=(0.9, 0.98),
                              eps=1e-6,
                              weight_decay=self.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, steps_per_epoch=self.steps_per_epoch, epochs=self.max_epochs, pct_start=0.05)
        scheduler = {"scheduler": self.scheduler, 
                     "interval": "step" }  # necessary to make PL call the scheduler.step after every batch
        return [optimizer], [scheduler]
    

    def calc_loss(self, inputs, targets, hiddens=None, lens=None):
        # pred
        if hiddens is None:
            preds = self(inputs, lens=lens)
        else:
            preds = self(inputs, hiddens=hiddens, lens=lens)
        # calc loss
        loss = self.loss_func(preds, targets)
        return loss, preds
    
    def preprocess(self, pat: torch.Tensor):
        """Inputs
            pats: A single patient sequence of shape [T, N] - T=steps, N=features. Unnormalized, contains NaNs
        """
        return self.data_module.preprocess(pat)

    
class LitCLIP(LitSeqModel):
    def __init__(self, *args, 
                 clip_name="ViT-B/16",
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.clip_name = clip_name
        
        import clip
        clip_model, transform = clip.load(clip_name, device="cpu", jit=False)
        
        # get relevant clip layers
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        # freeze attention layers
        for n, p in self.transformer.named_parameters():
            p.requires_grad = "mlp" in n or "ln" in n
        
        # define own mapping layers
        self.input_mapping = torch.nn.Linear(self.num_recurrent_inputs + self.num_static_inputs, self.token_embedding.weight.shape[1])
        self.out_mapping = torch.nn.Linear(self.transformer.width, 1)
        
    def make_preds(self, x, lens=None):
        # x = [BS, seq_len, feats]
        
        #x = self.token_embedding(text)#.type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = self.input_mapping(x)
        
        x = x + self.positional_embedding#.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) #.type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence) if we only want a single embedding for the sentence
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        #x = x[torch.arange(x.shape[0]), lens] @ self.text_projection
        
        # make prediction for each time-step
        x = self.out_mapping(x)
        return x
        
        
        
def load_gpt_model(name):
    device = torch.device("cpu")
    if name == "neo1.3":
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", pad_token_id=gpt_tokenizer.eos_token_id)
    elif name == "neo2.7":
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", pad_token_id=gpt_tokenizer.eos_token_id)
    elif name == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt_model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=gpt_tokenizer.eos_token_id)     
           
        
    gpt_model = gpt_model.to(device)
    return gpt_model, gpt_tokenizer


def apply_train_mode_gpt(mode, model):
    # freeze certain layers
    if mode == "freeze":
        for p in model.parameters():
            p.requires_grad = False
    elif mode == "train_norm":
        for n, p in model.named_parameters():
            p.requires_grad = "ln_" in n
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "train_mlp_norm":
        for n, p in model.named_parameters():
            p.requires_grad = "ln_" in n or "mlp" in n
    elif mode == "adapters":
        for n, p in model.named_parameters():
            p.requires_grad = "adapter" in n or "ln_" in n
        #    if p.requires_grad:
        #        print(n)
        #    #p.requires_grad = "ln_" in n or "mlp" in n

        
class LitGPT(LitSeqModel):
    def __init__(self, *args, 
                 gpt_name="gpt2",
                 mode="train_mlp_norm",
                 pretrained=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gpt_name = gpt_name
        # load model
        self.model, self.tokenizer = load_gpt_model(gpt_name)
        # re-init weights if not using pretrained
        if not pretrained:
            self.model.init_weights()  #.apply(self.model._init_weights)
        if mode == "adapters":
            self.model.add_adapter("icp", config=None, overwrite_ok=False, set_active=True)
        # freeze some params
        apply_train_mode_gpt(mode, self.model)
        # get width
        self.width = self.model.transformer.wte.weight.shape[1]
        # create input and output layers
        self.input_mapping = torch.nn.Linear(self.num_recurrent_inputs + self.num_static_inputs, self.width)
        self.out_mapping = torch.nn.Linear(self.width, 1)
        # replace output layer by our newly initialized one
        #model.model.transformer.wte = in_mapping
        self.model.lm_head = self.out_mapping
        
    def make_preds(self, x, lens=None):
        # x = [BS, seq_len, feats]
        x = self.input_mapping(x)
        x = self.model(inputs_embeds=x)["logits"]
        return x

        
class LitMLP(LitSeqModel):
    def __init__(self, 
                 data_module, 
                 hidden_size=256, 
                 dropout_val=0.2, 
                 **kwargs):
        super().__init__(data_module, **kwargs)
        self.save_hyperparameters()    
            
        self.hidden_size = hidden_size
        self.out_size = 1
        
        # define model
        # static part
        if self.static_idcs is not None:
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
        
        
    def make_preds(self, 
                x: torch.Tensor, 
                lens: Optional[torch.Tensor] = None, 
                hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # encode recurrence using mlp
        recurrent_stream = x[..., self.recurrent_idcs]
        recurrent_stream = self.encode_recurrent(recurrent_stream)
        
        # encode static inputs
        if self.static_idcs is not None:
            static_stream = x[..., self.static_idcs]
            static_stream = self.encode_static(static_stream)
        else:
            static_stream = 0
            
        # apply mlp to transform        
        x = self.out_mlp(static_stream + recurrent_stream)
        return x


class LitRNN(LitSeqModel):
    def __init__(self, data_module, hidden_size=256, dropout_val=0.2, rnn_layers=1,  rnn_type="lstm", **kwargs):
        super().__init__(data_module, **kwargs)
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.out_size = 1
        
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
        if self.static_idcs is not None:
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
        
    def make_preds(self, 
                x: torch.Tensor, 
                lens: Optional[torch.Tensor] = None, 
                hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        
        # encode recurrence using rnn
        recurrent_stream = x[..., self.recurrent_idcs]
        if lens is not None:
            recurrent_stream = pack_padded_sequence(recurrent_stream, lens.cpu(), batch_first=True, enforce_sorted=False)
        recurrent_stream, self.hiddens = self.rnn(recurrent_stream)
        if lens is not None:
            recurrent_stream, lens = pad_packed_sequence(recurrent_stream, batch_first=True)
        #recurrent_stream = self.layer_norm(recurrent_stream)
        
        # encode static inputs
        if self.static_idcs is not None:
            static_stream = x[..., self.static_idcs]
            static_stream = self.encode_static(static_stream)
        else:
            static_stream = 0
            
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
        
    def make_preds(self, x, hiddens=None, lens=None):
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

    def make_preds(self, x, hiddens=None, lens=None):
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
    def __init__(self, regression, use_macro_loss, pos_weight, use_huber):
        super().__init__()
        self.regression = regression
        self.use_macro_loss = use_macro_loss
        self.pos_weight = pos_weight
        
        if regression:
            if use_huber:
                self.loss_func = torch.nn.SmoothL1Loss(reduction='mean')
            else:
                self.loss_func = torch.nn.MSELoss(reduction='mean')
        else:
            self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        """Calculates the loss and considers missing values in the loss given by mask indicating where targets are NaN"""
        # shape: [BS, LEN, FEATS]
        mask = ~torch.isnan(target)
        
        if self.use_macro_loss:
            # Apply mask:
            num_feats = target.shape[-1]
            pred_per_pat = [pred_p[mask_p].reshape(-1, num_feats) for pred_p, mask_p in zip(pred, mask) if sum(mask_p) > 0]
            target_per_pat = [target_p[mask_p].reshape(-1, num_feats) for target_p, mask_p in zip(target, mask) if sum(mask_p) > 0]
            # calc raw loss per patient
            loss_per_pat = [self.loss_func(p, t) for p, t in zip(pred_per_pat, target_per_pat)]
            loss = torch.stack(loss_per_pat).mean()

            #if torch.isinf(loss) or torch.isnan(loss):
            #    print("Found NAN or inf loss!")
            #    print(loss)
            #    raise ValueError("Found NAN or inf!")
        else:
            #print(pred.shape, target.shape, mask.shape)
            loss = self.loss_func(pred[mask], target[mask])

        return loss
