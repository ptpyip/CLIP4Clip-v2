import os
from typing import Optional
from functools import partial

import torch
import torch.distributed
from torch import nn
from pydantic import BaseModel

from .modules import CrossEn, AllGather
from .pretrainedCLIP import PreTrainedClip

from . import temporal 
from .temporal import TemporalMode, TemporalTransformer, BaseTemporalModule

MODELS = [
    "meanP-ViT-B/16","meanP-ViT-B/32",
    # "maxP-ViT-B/16","maxP-ViT-B/32",
    # "Trans-ViT-B/16","Trans-ViT-B/32"
]


class CLIPConfig(BaseModel):
    name: str
    pretrained: bool = True
    freeze_layer_num: int
  
class ModelConfig(BaseModel):
    name: str
    # clip: CLIPConfig
    # image_resolution: int
    # context_length: int
    temporal_mode: str
    ckpt_path: Optional[str] = None
    
    

def build_model(config: ModelConfig, state_dict: dict = {}, world_size=None, rank=None):
    """ build model from a given state dict
    if state_dict is {} => build from empty
    """
    assert config.name in MODELS
        
    
    model = CLIP4Clip(config.name, config.temporal_mode, world_size=world_size, rank=rank)    #type: ignore
    if state_dict != {}:
        model.load_state_dict(state_dict)
    return model.float()
    
    
class CLIP4Clip(PreTrainedClip):
    CLIP_NAME = "ViT-B/32"
    def __init__(self,
        name,
        temporal_mode: TemporalMode = TemporalMode.MEAN_POOLING,
        hidden_size = 512,
        num_temporal_hidden_layers = 2,
        max_temporal_embeddings = 128,
        distributed=False, world_size=None, rank=None
    ) -> None:

        clip_name = name.split("-", 1)[1]
        super(CLIP4Clip, self).__init__(clip_name)                                ## init clip
        self.input_resolution = self.clip.visual.input_resolution               # for transform(): Image -> Tensor, with right resolution
        self.max_num_frame = self.clip.context_length                           # ensure input not exceed temporal context length
       
        self.temporal_mode = temporal_mode 
        self.hidden_size = hidden_size
        self.num_temporal_hidden_layers = num_temporal_hidden_layers
        
        self.temporal = self._init_temporal(max_temporal_embeddings)
        # self.temporal_trans = None
        # if temporal_mode == TemporalMode.TRANSFORMER:
        #     self.temporal_trans = self._init_temporal(max_temporal_embeddings)
        #     assert self.num_temporal_embeddings <= max_temporal_embeddings
        
        self.loss_fn = CrossEn()
        self.norm = lambda x: x / x.norm(dim=-1, keepdim=True)
        
        self.allgather = lambda tensor: AllGather.apply(tensor, world_size, rank)
    
        return
    

    def forward_text(self, text):
        bs = text.size(0)
        
        text_feature = self.clip.encode_text(text).float()
        text_feature = text_feature.view(
            bs, -1, text_feature.size(-1)
        ).squeeze(1)
        
        return  text_feature
    
    
    def forward_visual(self, frames, video_mask):
        """
        here in og CLIP4Clip implementation they apply layernorm and projection to entire tans output.
        """
        bs = video_mask.size(0)
        frames = self.clip.encode_image(frames).float()
        frames = frames.view(
            bs, -1, frames.size(-1)
        )
        
        temporal_feature = self.temporal(frames, video_mask)
        # return self.norm(temporal_feature) if self.training else temporal_feature
        return temporal_feature
    
    def forward(self, text, video, video_mask) -> torch.Tensor:
        text = text.view(-1, text.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # bs x L x 3 x H x W
        video = torch.as_tensor(video).float()
        bs, L, channel, h, w = video.shape
        video = video.view(bs*L, channel, h, w)
        
        text_feature = self.forward_text(text)
        video_feature = self.forward_visual(video, video_mask)
        
        ## blocking
        text_feature = self.allgather(text_feature)
        video_feature = self.allgather(video_feature)
        video_mask = self.allgather(video_mask)
        torch.distributed.barrier()
               
        text_feature = self.norm(text_feature)
        video_feature = self.norm(video_feature)  
        sim_matrix, *_tmp = self.get_similarity_logits(
            text_feature, video_feature
        )
        
        sim_loss1 = self.loss_fn(sim_matrix)
        sim_loss2 = self.loss_fn(sim_matrix.T)
        loss = (sim_loss1 + sim_loss2) / 2

        return loss
    
    def get_similarity_logits(self, text_feature, video_feature):
        # if shaped is False:
        #     attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        #     video_mask = video_mask.view(-1, video_mask.shape[-1])

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(text_feature, video_feature.T)
        return retrieve_logits
    
    def _init_temporal(self, max_temporal_embeddings) -> BaseTemporalModule:         
        if self.temporal_mode is TemporalMode.MEAN_POOLING:
            return temporal.MeanPooling()
                    
        self.num_temporal_embeddings = self.clip.positional_embedding.shape[0]
        self.transformer_width = self.clip.ln_final.weight.shape[0]
        heads = self.transformer_width // 64
        
        assert self.num_temporal_embeddings <= max_temporal_embeddings
        temporal_trans = TemporalTransformer(
            width=self.transformer_width, 
            layers=self.num_temporal_hidden_layers,
            heads=heads, 
            hidden_size=self.hidden_size,
            num_temporal_embeddings=self.num_temporal_embeddings
        )   
       
        temporal_state_dict = {
            "temporal_embeddings.weight": self.clip.positional_embedding.clone()
        } 
        
        for key, val in self.clip.transformer.state_dict(prefix="transformer.").items():
            num_layer = int(key.split(".")[2]) 
            if num_layer >= self.num_temporal_hidden_layers:
                break
            
            temporal_state_dict[key] = val.clone() 
        
        temporal_trans.load_state_dict(temporal_state_dict)
        return temporal_trans