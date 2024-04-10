import torch
from torch import nn 

from abc import ABC, abstractmethod
from enum import Enum

from .transformer import TransformerEncoder

class TemporalMode(Enum):
    MEAN_POOLING = "meanP"
    TRANSFORMER = "seqTrans"
    # CrossTransformer = 2        # not used.
    
class BaseTemporalModule(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def forward(self, src, mask):
        raise NotImplementedError
    
# class MeanPooling(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
        
#         self.norm = lambda x: x / x.norm(dim=-1, keepdim=True)
        
#     def forward(self, x, video_mask):
#         x = self.norm(x) if self.training else x
        
#         video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
#         x = x * video_mask_un
        
#         video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
#         video_mask_un_sum[video_mask_un_sum == 0.] = 1.
#         out = torch.sum(x, dim=1) / video_mask_un_sum
        
#         return out

class TemporalTransformer(BaseTemporalModule):
    def __init__(self, 
        width: int, 
        layers: int, 
        heads: int,
        hidden_size: int,
        num_temporal_embeddings:int, 
        mean_pool=True
    ):
        super().__init__()
        self.temporal_embeddings = nn.Embedding(
            num_temporal_embeddings, hidden_size
        )
        # self.transformer == nn.TransformerEncoder
        self.transformer = TransformerEncoder(width, layers, heads)
        self.pooling = MeanPooling() if mean_pool else lambda x, _: x[:, 0, 1]
        
    def forward(self, x: torch.Tensor, video_mask):
        seq_length = x.size(1)
        
        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        
        temporal_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        temporal_embeddings = self.temporal_embeddings(
            temporal_ids.unsqueeze(0).expand(x.size(0), -1)
        )
        
        temp = x + temporal_embeddings
        out = self.transformer(
            temp.permute(1, 0, 2),              # NLD -> LND
            extended_video_mask
        ).permute(1, 0, 2)                      # LND -> NLD
        
        return self.pooling(out + x, video_mask) 
    

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
    
class MeanPooling(BaseTemporalModule):
    """perform Global Average Pooling"""
    def __init__(self):
        super().__init__()
        
        self.norm = lambda x: x / x.norm(dim=-1, keepdim=True)

    def forward(self, out, mask) -> torch.Tensor:
        out = self.norm(out) if self.training else out
        
        mask_un = mask.to(dtype=torch.float).unsqueeze(-1)
        out = out * mask_un
        
        mask_un_sum = torch.sum(mask_un, dim=1, dtype=torch.float)
        mask_un_sum[mask_un_sum == 0.] = 1.         # avoid divide zero
        pooled_out = torch.sum(out, dim=1) / mask_un_sum
        
        return pooled_out


class MaxPooling(BaseTemporalModule):
    """perform Global Max Pooling"""
    def __init__(self):
        super().__init__()
        
        self.norm = lambda x: x / x.norm(dim=-1, keepdim=True)

    def forward(self, out, mask) -> torch.Tensor:
        out = self.norm(out) if self.training else out
        
        mask_un = mask.to(dtype=torch.float).unsqueeze(-1)
        pooled_out = torch.max(out * mask_un, dim=1) 
        
        return pooled_out