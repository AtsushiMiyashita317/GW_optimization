import torch
from torch.nn.functional import pad

import sparse


def gw_algebra(w:torch.Tensor):
    """Get generalized apw lie algebra L(w)

    Args:
        w (ndarray, shape=(...,dim-1)): warping parameter
        
    Returns:
        L (ndarray, shape=(...,dim,dim)): generalized apw lie algebra L(w)
    """
    dim = w.shape[-1]+1
    # _w = _w/torch.arange(1,dim,device=device)
    w = torch.cat([w.flip([-1]),torch.zeros_like(w[...,:1]),-w],dim=-1)
    L = torch.as_strided(
            w,
            size=w.shape[:-1]+(dim,dim),
            stride=w.stride()[:-1]+(w.stride(-1),w.stride(-1))
        ).flip([-1]).mul(
            torch.arange(-dim//2+1,dim//2+1,device=w.device)
        )
    return L

def normalize(x:torch.Tensor):
    return x/x.abs()

def gw_matrix(w:torch.Tensor,n_in:int,n_out:int,neighbor=5):
    size = w.size(-1)
    z = normalize(
            torch.fft.fft(
                pad(
                    torch.matrix_exp(
                        gw_algebra(w)
                    )[...,:,size//2+1],
                    (2*n_out-size-1,0)
                ).roll(size//2+1,-1),
                dim=-1
            )[...,:n_out]
        )
    A = sparse.sparse_warp(z,n_in,neighbor)
    return A
