import torch

from GAPW.modules import functional

class GAPW(torch.nn.Module):
    def __init__(self,in_channels:int,order:int,device=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.w = torch.nn.Parameter(
            torch.empty(1,in_channels,order,device=device,dtype=torch.float))
        self.reset_parameters()        
        
    def forward(self, x):
        x = functional.gapw(x,w=self.w,dim=-1)
        return x
        
    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.w)
        