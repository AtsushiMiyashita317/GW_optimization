import torch
from torch.nn.functional import pad

def transpose(x:torch.Tensor,middle:int,back:int):
    if type(x) is not torch.Tensor: return x
    if back == 0: return x
    dims = tuple(range(x.ndim))
    dims = dims[:middle]+dims[back:]+dims[middle:back]
    return torch.permute(x,dims=dims)

def add_dim(*argndim,resultndim=None,**kwargndim):
    def _add_dim(func):
        def wrapper(*args,dim=None,**kwargs):
            if dim is not None:
                assert dim < 0, dim
                dim += 1
                _args = [None]*len(args)
                _kwargs = {}
                for i in range(len(args)):
                    _args[i] = transpose(args[i],dim-argndim[i],dim)
                for key in kwargs.keys():
                    _kwargs[key] = transpose(kwargs[key],dim-kwargndim[key],dim)
                result = func(*_args,**_kwargs)
                if type(result) is torch.Tensor:
                    result = transpose(result,dim-resultndim,-resultndim)
                if type(result) is tuple:
                    for i in range(len(result)):
                        result[i] = transpose(result[i],dim-resultndim[i],-resultndim[i])
                return result
            else:
                return func(*args,**kwargs)
        return wrapper
    return _add_dim

@add_dim(1,w=1,resultndim=2)
def gapw_algebra(w:torch.Tensor):
    """Get generalized apw lie algebra L(w)

    Args:
        w (ndarray, shape=(...,dim-1)): warping parameter
        
    Returns:
        la (ndarray, shape=(...,dim,dim)): generalized apw lie algebra L(w)
    """
    device = w.device
    dim = w.shape[-1]+1
    _w = w
    _w = _w/torch.arange(1,dim,device=device)
    _w = torch.cat([torch.flip(_w,dims=[-1]),torch.zeros(_w.shape[:-1]+(1,),device=device),-_w],dim=-1,)
    la = torch.clone(
            torch.as_strided(_w,
                size=_w.shape[:-1]+(dim,dim),
                stride=_w.stride()[:-1]+(_w.stride(-1),_w.stride(-1))
                ).flip([-1]))
    la *= torch.arange(-dim//2+1,dim//2+1,device=device)
    return la

@add_dim(1,w=1,resultndim=2)
def gapw_adjoint(w:torch.Tensor):
    """Get generalized apw adjoint matrix Ad(L(w))

    Args:
        w (ndarray, shape=(...,dim-1)): warping parameter
        dim (int, optional): dimension of output matrix
    
    Returns:
        ad (ndarray, shape=(...,dim,dim)): generalized apw adjoint matrix Ad(L(w))
    """
    device = w.device
    dim = w.shape[-1]
    _w = w
    # _w = _w/torch.arange(1,dim+1,device=device)
    _w = torch.cat([
        -_w.flip([-1]),
        torch.zeros(_w.shape[:-1]+(1,),device=device),
        _w,
        torch.zeros(_w.shape[:-1]+(dim,),device=device)
        ],dim=-1,)
    coef = torch.arange(-2*dim+1,3*dim+1,device=device)
    ad = torch.zeros(_w.shape[:-1]+(dim,dim),device=device)
    ad += (
          torch.as_strided(coef,
            size=(dim,dim),
            stride=(coef.stride(-1),2*coef.stride(-1)))* \
          torch.as_strided(_w,
            size=_w.shape[:-1]+(dim,dim),
            stride=_w.stride()[:-1]+(_w.stride(-1),_w.stride(-1)),
            storage_offset=1)
          ).flip([-1])
    ad -= torch.as_strided(coef,
            size=(dim,dim),
            stride=(coef.stride(-1),2*coef.stride(-1)),
            storage_offset=2*dim+2)* \
          torch.as_strided(_w,
            size=_w.shape[:-1]+(dim,dim),
            stride=_w.stride()[:-1]+(_w.stride(-1),_w.stride(-1)),
            storage_offset=dim+2)
    return ad

@add_dim(1,0,w=1,size=0,resultndim=2)
def gapw_matrix(w:torch.Tensor,size:int=None):
    _w = w/torch.arange(1,w.shape[-1]+1,device=w.device)
    if size is not None and size-1 > w.shape[-1]:
        _w = pad(_w,(0,size-1-_w.shape[-1]))
    ga = gapw_algebra(_w)
    m = torch.matrix_exp(ga)
    size = m.shape[-1]
    centor = (size-1)//2
    tmp = m[...,centor:,centor:]
    tmp[...,1:1+centor] += torch.flip(m[...,centor:,:centor],[-1])
    m = tmp
    return m

@add_dim(1,1,2,0,s=1,w=1,m=2,output_dim=0,resultndim=1)
def gapw(s:torch.Tensor,w:torch.Tensor=None,m:torch.Tensor=None,output_dim=None):
    """apply general warping to inputs

    Args:
        s (np.ndarray,shape=(...,length)): input signal to warp
        w (np.ndarray,shape=(...,dim-1)): warping parameter
        m (np.ndarray,shape=(...,dim-1,dim-1)): warping matrix
        
    Returns:
        _s (np.ndarray,shape=(...,length)): warped signal
        
    Note:
        Either w or m must be specified.
        The rest shape represented by [...] must be brordcastable.
    """
    assert not ((w is None) and (m is None)),'Either t or m must be specified.'
    assert (w is None) or (m is None),'Either t or m must be specified.'
    _s,_w,_m = s,w,m
    
    # convert complex to real
    if s.dtype is torch.cfloat:
        _s = torch.view_as_real(_s)
    else:
        _s = _s[...,None]
    
    # prepare warping matrix
    input_dim = _s.shape[-2]
    output_dim = input_dim if output_dim is None else output_dim
    size = (max(input_dim, output_dim)-1)*2
    if _w is not None:
        _m = gapw_matrix(_w,size=size)
    _m = _m[...,:output_dim,:input_dim]
    
    # warping
    f = torch.fft.rfft(torch.cat([_s,torch.flip(_s,[-2])[...,1:-1,:]],dim=-2),dim=-2,norm='forward').real
    f = _m@f
    _s = torch.fft.rfft(torch.cat([f,torch.flip(f,[-2])[...,1:-1,:]],dim=-2),dim=-2,norm='backward').real
    
    # convert real to complex   
    if s.dtype is torch.cfloat:
        _s = _s.real.contiguous()
        _s = torch.view_as_complex(_s)
    else:
        _s = _s[...,0].real
         
    return _s


if __name__=='__main__':
    from matplotlib import pyplot as plt
    
    # print gapw_algebra coefficient
    w = torch.ones(9)
    al = gapw_algebra(w)
    print('gapw_algebra coefficient')
    print(al)
    # print gapw_adjoint coefficient
    w = torch.ones(9)
    ad = gapw_adjoint(w)
    print('gapw_adjoint coefficient')
    print(ad)
    # test gapw
    plt.figure()
    # even length
    x = torch.linspace(0,1,256)
    w = torch.randn(16)*0.1
    y = gapw(x, w=w)
    plt.plot(x, y, label='even')
    # odd length
    x = torch.linspace(0,1,257)
    w = torch.randn(16)*0.1
    y = gapw(x, w=w)
    plt.plot(x, y, label='odd')
    plt.savefig('./test1_functional.png')
    # batch
    plt.figure()
    x = torch.linspace(0,1,256)
    w = torch.randn(10,16)*0.1
    y = gapw(x, w=w)
    plt.plot(x, y.T)
    plt.savefig('./test2_functional.png')
    # test add_dim
    plt.figure()
    x = torch.linspace(0,1,256)
    w = torch.randn(16,10)*0.1
    y = gapw(x[:,None], w=w, dim=-2)
    plt.plot(x, y)
    plt.savefig('./test3_functional.png')
    # plot map
    plt.figure()
    x = torch.eye(256)
    w = torch.randn(16)*0.1
    y = gapw(x, w=w)
    plt.imshow(y, origin='lower')
    plt.savefig('./test4_functional.png')
    # specify output_dim
    plt.figure()
    x = torch.eye(256)
    w = torch.randn(16)*0.1
    y = gapw(x, w=w, output_dim=128)
    plt.imshow(y, origin='lower')
    plt.savefig('./test5_functional.png')
    plt.figure()
    x = torch.eye(256)
    w = torch.randn(16)*0.1
    y = gapw(x, w=w, output_dim=512)
    plt.imshow(y, origin='lower')
    plt.savefig('./test6_functional.png')
    # test for attention
    batch = 10
    head = 5
    time1 = 64
    time2 = 128
    d_k = 32
    order = 16
    w = torch.randn(batch, head, order)*0.1
    e = torch.eye(time2)
    attn = gapw(e, w=w[...,None], output_dim=time1, dim=-2)
    print(attn.shape)
    plt.figure()
    plt.imshow(attn[0,0], origin='lower')
    plt.savefig('./test7_functional.png')