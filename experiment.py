import random
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import torch
import torchaudio

from GAPW.modules import functional as F

def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def random_warping_path(length):
    d = torch.rand(20)[:,None]+0.5*torch.rand((length+19)//20)
    d = d.ravel()[:length].cumsum(0)
    path = d/d[-1]*length
    return path

def linear_interpolation(x:torch.Tensor,idx:torch.Tensor):
    i = torch.minimum(torch.tensor(x.size(-1)-1),idx.floor().long())
    j = torch.minimum(torch.tensor(x.size(-1)-1),i+1)
    f = idx-i
    return (1-f)*x[...,i]+f*x[...,j]

def training(pathes, specs, method):
    error = torch.zeros(10,10)
    losses = torch.zeros(10,500)

    for i in range(10):
        print(method, f'{i+1}/10')
        bar = tqdm(total=500)
        path = pathes[i]
        specs_warp = linear_interpolation(specs,path)
        
        w = torch.nn.parameter.Parameter(torch.zeros(10,1,1 if method=='APW' else 256, device='cuda'))
        optimizer = torch.optim.Adam([w])
        
        for k in range(100):
            perm = torch.randperm(specs.shape[-2])
            x = specs[:,perm]
            y = specs_warp[:,perm]
            loss_sum = 0.
            for l in range(specs.shape[-2]//4):
                y_ = F.gapw(x[:,l*4:(l+1)*4], w)
                loss = (y_-y[:,l*4:(l+1)*4]).square().sum()
                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses[i,k] = loss_sum
            bar.update(1)
        
        for k in range(400):
            specs_ = F.gapw(specs, w)
            loss = (specs_-specs_warp).square().sum()
            losses[i,k+100] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.update(1)
            
        t = torch.arange(specs.shape[-1])[None]
        pred = F.gapw(t.cuda(),w.detach())[:,0]
        error[i] = (pred-path).abs().mean(-1)/path.shape[0]
    
    return error, losses


if __name__=='__main__':
    torch_fix_seed()
    
    signs = []
    for i in range(10):
        sign,sr = torchaudio.load(f'./audio/sample{i}.wav')
        signs.append(sign[0])
    length = min([sign.shape[0] for sign in signs])
    signs = torch.stack([sign[:length] for sign in signs])
    specs = torch.stft(signs,256,128,return_complex=True).abs().cuda()

    gts = torch.zeros(10,specs.shape[-1])
    for i in range(10):
        gts[i] = random_warping_path(specs.shape[-1])
    gts = gts.cuda()

    error_gw, losses_gw = training(gts, specs, 'GW')
    error_apw, losses_apw = training(gts, specs, 'APW')
        
    plt.figure(figsize=(4,3))
    plt.plot(losses_gw.mean(0),label='GW')
    plt.plot(losses_apw.mean(0),label='APW')
    plt.xlabel('Epoch')
    plt.ylabel('L2 loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./result/fig8.png')
    plt.show()

    plt.figure(figsize=(6/1.3,2/1.3))
    plt.boxplot([error_gw.detach().ravel(),error_apw.detach().ravel()], vert=False, labels=['GW','APW'], showfliers=False, widths=0.4)
    plt.xlabel('Warping Error')
    plt.tight_layout()
    plt.savefig('./result/fig9.png')
    plt.show()
