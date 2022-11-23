import torch
import numpy as np


class ContinuousEmbedding(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        self.weights = torch.nn.Parameter(torch.zeros(1, 1, in_size, out_size).normal_(0, 0.1))
        self.bias = torch.nn.Parameter(torch.rand(1, 1, out_size))

    def forward(self, x):
        x = x.unsqueeze(-1)  # add dim to enable multiplication 
        out = x * self.weights + self.bias
        return out
    

class NanEmbed(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        # create embedding weights
        self.cont_emb = ContinuousEmbedding(in_size, out_size)
        
    def forward(self, x):
        # create mask to later fill with zeros
        mask = torch.isnan(x)
        x = torch.nan_to_num(x)
        # embed each feature into a larger embedding vector of size out_size
        out = self.cont_emb(x)
        # shape [batch size, seq_len, in_size, out_size]
        # fill embedding with 0 where we had a NaN before
        with torch.no_grad():
            out[mask] = 0
            
        # average the embedding
        emb = out.mean(dim=2) 
        return emb
    
    
class NanEmbedTransformer(torch.nn.Module):
    def __init__(self, in_size, out_size,
                n_layers=3, n_heads=4, dropout=0.2):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        # create embedding weights
        self.cont_emb = ContinuousEmbedding(in_size, out_size)
        # create transformer that operates on these
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        encoder_layers = TransformerEncoderLayer(out_size, n_heads, out_size * 4, 
                                                 dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        
        scale = out_size ** -0.5
        self.positional_embedding = torch.nn.Parameter(scale * torch.randn(in_size, out_size))
        self.ln_pre = torch.nn.LayerNorm(out_size)
        
    def forward(self, x):
        # create mask to later fill with zeros
        mask = torch.isnan(x)
        x = torch.nan_to_num(x)
        # embed each feature into a larger embedding vector of size out_size
        out = self.cont_emb(x)
        # shape [batch size, seq_len, in_size, out_size]
        # fill embedding with 0 where we had a NaN before
        with torch.no_grad():
            out[mask] = 0
        # apply transformer to "tokens"
        # transform to shape  [batch size * seq_len, in_size, out_size]
        orig_shape = out.shape
        out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])
        # first add positional encoding
        out = out + self.positional_embedding.to(out.dtype)
        # transform
        out = self.ln_pre(out)
        #dtype = self.transformer_encoder.layers[0].self_attn.out_proj.weight.dtype
        #print(dtype, out.dtype)
        #out = out.to(dtype)
        mask = torch.zeros(out.shape[1], out.shape[1], dtype=out.dtype, device=out.device)
        out = self.transformer_encoder(out, mask)
        # put back into original sequence shape
        out = out.reshape(*orig_shape)
        # average the embedding over all tokens
        emb = out.mean(dim=2) 
        return emb
