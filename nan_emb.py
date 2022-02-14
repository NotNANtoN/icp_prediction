import torch
import numpy as np


class ContinuousEmbedding(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        mask = torch.ones(in_size, in_size * out_size).bool()
        for i in range(in_size):
            mask[i, i * out_size: i * out_size + out_size] = False
        self.register_buffer("mask", mask)
        
        self.weights = torch.nn.Parameter(torch.zeros(in_size, in_size * out_size).normal_(0.01))
        with torch.no_grad():
            self.weights[mask] = 0
        self.bias = torch.nn.Parameter(torch.rand(in_size * out_size))

    def forward(self, x):
        weights = self.weights * self.mask
        out = (x @ weights) + self.bias
        return out.reshape(-1, self.in_size, self.out_size)


class NanEmbed(torch.nn.Module):
    def __init__(self, in_size, out_size, use_conv=True):
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
        # shape [batch size, in_size, out_size]
        # fill embedding with 0 where we had a NaN before
        #repeated_mask = mask.unsqueeze(-1).repeat(1, 1, self.out_size)
        with torch.no_grad():
            out[mask] = 0
        # average the embedding
        emb = out.mean(dim=1) 
        return emb



class NanEmbedWeird(torch.nn.Module):
    def __init__(self, in_size, out_size, use_conv=True):
        # Create filterset for each sample
        
        
        
        weights = []
        for _ in range(N):
            #weight = nn.Parameter(torch.randn(15, 3, 5, 5))
            weight = nn.Parameter(torch.randn(out_size, in_size, 1))
            weights.append(weight)
        self.weights = torch.stack(weights)

    
    def manual(self, x):
        # Apply manually
        outputs = []
        for idx in range(N):
            input = x[idx:idx+1]
            weight = weights[idx]
            output = F.conv2d(input, weight, stride=1, padding=2)
            outputs.append(output)

        outputs = torch.stack(outputs)
        outputs = outputs.squeeze(1) # remove fake batch dimension
        return outputs
        #print(outputs.shape)
        #> torch.Size([10, 15, 24, 24])

    def forward(self, x):
        # N, C, H, W = 10, 3, 24, 24 = x
        orig_shape = x.shape
        bs = orig_shape[0]
        feats = orig_shape[-1]
        x = x.reshape(-1, feats)
        
        
        # Use grouped approach
        weights = self.weights.view(-1, self.in_size, 1)
        #print(weights.shape)
        #> torch.Size([150, 3, 5, 5])
        # move batch dim into channels
        x = x.view(1, -1, feats)
        #print(x.shape)
        #> torch.Size([1, 30, 24, 24]) - > (1, bs * out_size, feat)
        # Apply grouped conv
        #outputs_grouped = F.conv2d(x, weights, stride=1, padding=2, groups=N)
        outputs_grouped = F.conv2d(x, weights, stride=1, padding=0, groups=bs)
        outputs_grouped = outputs_grouped.view(bs, self.out_size, feats)

class NanEmbedOld(torch.nn.Module):
    def __init__(self, in_size, out_size, use_conv=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_conv = use_conv
        # create embedding weights
        #if use_conv:
        #    self.emb_layer = torch.nn.Conv1d(1, out_size, kernel_size=1)
        #else:
        self.emb_layers = torch.nn.ModuleList([torch.nn.Linear(1, out_size) for _ in range(in_size)])
        
        #conv_in = inputs.reshape(4 * 75, -1, 1).cuda()

        
        #conv = torch.nn.Conv1d(in_size, in_size * out_size, 1, stride=1, padding=0,
        #               groups=in_size, bias=True)
        
    def forward(self, x):
        orig_shape = x.shape
        bs = orig_shape[0]
        feats = orig_shape[-1]
        x = x.reshape(-1, feats)
        
        #if self.use_conv:
        #    out = self.emb_layer(x.unsqueeze(1))
        #else:
        out = torch.stack([layer(x[:, i].unsqueeze(1)) for i, layer in enumerate(self.emb_layers)], dim=-1)
        # select all the embeddings of non-nan inputs and sum them
        # is done in a loop per batch because the NaN mask differs per batch part, so unequal shapes
        
        #with torch.no_grad():
        #    out[torch.isnan(out)] = 0
        out = torch.nan_to_num(out)

        emb = out.mean(dim=-1)
        
        #mask = torch.isnan(x)
        #bs = x.shape[0]
        #emb = torch.stack([out[i][:, torch.where(mask[i])[0]].sum(dim=-1) / self.in_size
        #                    for i in range(bs)])
        
        #emb = emb.reshape(*orig_shape[:-1], self.out_size)
        return emb
    

        
        
if __name__ == "__main__":        
    data1 = torch.arange(0, 10).float()
    data1[[1, 5]] = np.nan
    data1[-1] = 0
    data2 = data1.clone()
    data2[0] = np.nan
    data = torch.stack([data1, data2])
    print("Data: ", data)
    
    print("Conv emb: ")
    emb_conv = NanEmbed(10, 16, use_conv=True)
    out = emb_conv(data)
    print(out.shape, out.mean(), out.std())
    print(out)
    #print("Weights: ")
    #for n, p in emb_conv.named_parameters():
    #    print(n, p.shape, p)
    print()
    
    print("Lin emb: ")
    emb_lin = NanEmbed(10, 16, use_conv=False)
    out = emb_lin(data)
    print(out.shape, out.mean(), out.std())
    print(out)
    #print("Weights: ")
    #for n, p in emb_lin.named_parameters():
    #    print(n, p.shape, p)
    print()
    
    quit()
    
        
        
        
        
        
    data = torch.rand(1, 10, 1)
    data[:, [0, 5]] = np.nan

    print(data)

    lin = torch.nn.Linear(10, 16)
    print(lin(data.squeeze(-1)))


    conv = torch.nn.Conv1d(1, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    #print(conv(data))



