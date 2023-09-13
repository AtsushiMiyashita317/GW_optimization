import torch

class sparse_bmm_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx:torch.autograd.function.FunctionCtx,w:torch.sparse.Tensor,x:torch.Tensor):
        ctx.save_for_backward(w,x)
        y = torch.bmm(w,x)
        return y
    
    @staticmethod
    def backward(ctx:torch.autograd.function.FunctionCtx,dy:torch.Tensor):
        w:torch.sparse.Tensor
        x:torch.Tensor
        w,x = ctx.saved_tensors
        
        dx = torch.bmm(w.transpose(-2,-1),dy)
        
        indices = w.indices()
        dw =    torch.sparse_coo_tensor(
                    indices=indices,
                    values=torch.sum(
                        dy[indices[0],indices[1]]*x[indices[0],indices[2]],
                        dim=-1
                    ),
                    size=w.size()
                )
        
        return dw, dx
        
sparse_bmm_internal = sparse_bmm_impl.apply

def sparse_bmm(w:torch.sparse.Tensor,x:torch.Tensor):
    y:torch.Tensor = sparse_bmm_internal(w,x)
    return y

def meshgrid(size,device=None)->torch.Tensor:
    ndim = len(size)
    return torch.stack([
        torch.arange(size[i],device=device).reshape([1]*i+[-1]+[1]*(ndim-i-1)).broadcast_to(size) for i in range(ndim)
    ],dim=0)
    
def pulse_ifft(z:torch.Tensor,N:int,eps=1e-4):
    r = ((z-1)*(1/z-1)*2*N).real
    z0 = z**N
    z1 = z0*z
    x0 = (z0+1/z0).real
    x1 = (z1+1/z1).real
    return (x0-x1-r)/(r+eps) + 1

def sparse_warp(z:torch.Tensor,N:int,n_neighbor:int):
    idx = -z.angle().div(torch.pi/N,rounding_mode='floor')
    idx = idx[...,None]+torch.arange(-n_neighbor,n_neighbor+1,device=z.device)
    
    values = pulse_ifft(z[...,None]*idx.mul(1j*torch.pi/N).exp(),N).flatten()
    
    indices = meshgrid(idx.size(),device=z.device)
    indices[-1] = idx
    indices = indices.flatten(start_dim=1)
    
    b = (0<=indices[-1]) & (indices[-1]<N)
    indices = indices[:,b]
    values = values[b]
    
    return torch.sparse_coo_tensor(indices=indices,values=values,size=list(z.shape)+[N])
    