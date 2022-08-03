import torch
import clip
import torchvision

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='densenet', help='model name')
parser.add_argument('--bs', type=int, default=None, help='batch size')
parser.add_argument('--adapter', type=int, default=0)

args = parser.parse_args()

if args.model == 'densenet':
    model = torchvision.models.densenet121(pretrained=True)
else:
    model = clip.load(args.model, jit=False)[0].visual


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() 


if args.adapter:
    from adapter_utils import add_adapters_visual
    model = add_adapters_visual(model, down_sample_size=256, adapter_flow="hard")


def measure_num_trainable(model):
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    return num_params

def measure_num_total(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params

max_batch_sizes = {"densenet": 128, "RN50": 128, "ViT-B/32": 128, 
                 "ViT-B/16": 64, "ViT-L/14": 8, "RN50x4": 48}
batch_size = max_batch_sizes[args.model] if args.bs is None else args.bs
#size = 256 if args.model == "densenet" else 224
if args.model == "densenet":
    size = 256
elif args.model == "RN50x4":
    size = 288
else: 
    size = 224

from ptflops import get_model_complexity_info

with torch.cuda.amp.autocast():
    with torch.cuda.device(0):
        model.cuda()
        macs, params = get_model_complexity_info(model, (3, size, size), as_strings=True,
                                                print_per_layer_stat=False, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


import time

def measure_throughput(model, batch_size, num_batches, size=224, backward=False):
    model.cuda()
    data = torch.randn(batch_size * num_batches, 3, size, size)
    #start = torch.cuda.Event(enable_timing=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    start = time.time()
    if not backward:
        model.eval()
        with torch.inference_mode():
            for i in range(num_batches):
                x = data[i * batch_size: (i + 1) * batch_size].cuda()
                with torch.cuda.amp.autocast():
                    y = model(x)
    else:
        model.train()
        for i in range(num_batches):
            x = data[i * batch_size: (i + 1) * batch_size].cuda()
            with torch.cuda.amp.autocast():
                y = model(x)
                loss = y.mean().float()
            opt.zero_grad()
            scaler.scale(loss).backward()
            convert_models_to_fp32(model)
            scaler.step(opt)
            scaler.update()
    #return start.elapsed_time(torch.cuda.Event(enable_timing=True)) / num_batches
    return 1 / ((time.time() - start) / num_batches) * batch_size



print('Number of parameters: {}'.format(measure_num_total(model)))
print('Number of trainable parameters: {}'.format(measure_num_trainable(model)))

torch.cuda.reset_peak_memory_stats(device=None)

print('Max batch size: {}'.format(batch_size))
print('Throughput inference: {}'.format(measure_throughput(model, batch_size, 10, size)))
print("Peak GPU usage inference: ", torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024)

torch.cuda.reset_peak_memory_stats(device=None)
print('Throughput with backward and opt step: {}'.format(measure_throughput(model, batch_size, 10, size, backward=True)))
print("Peak GPU usage training: ", torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024)
