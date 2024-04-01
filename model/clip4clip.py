import os

import torch
from torch import nn

from .modules import CrossEn
from .pretrainedCLIP import PreTrainedClip

from . import temporal 
from .temporal import TemporalMode, TemporalTransformer

MODELS = [
    "meanP-ViT-B/16","meanP-ViT-B/32",
    # "maxP-ViT-B/16","maxP-ViT-B/32",
    # "Trans-ViT-B/16","Trans-ViT-B/32"
]

def build_model(model_name, state_dict: dict = {}):
    """ build model from a given state dict
    if state_dict is {} => build from empty
    """
    assert model_name in MODELS
    
    model = CLIP4Clip(model_name)
    if state_dict != {}:
        model.load_state_dict(state_dict)
    return model.float()
    
    
class CLIP4Clip(PreTrainedClip):
    CLIP_NAME = "ViT-B/32"
    def __init__(self,
        name,
        temporal_mode: TemporalMode = TemporalMode.MEAN_POOLING,
        hidden_size = 512,
        num_temporal_hidden_layers = 4,
        max_temporal_embeddings = 128,
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
    
        return
    

    def forward_text(self, text):
        bs = text.size(0)
        
        text_feature = self.clip.encode_text(text).float()
        text_feature = text_feature.view(
            bs, -1, text_feature.size(-1)
        ).squeeze(1)
        
        return self.norm(text_feature) if self.training else text_feature
    
    
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
        return self.norm(temporal_feature) if self.training else temporal_feature
        
    
    def _init_temporal(self, max_temporal_embeddings) -> TemporalTransformer:         
        if self.temporal_mode is TemporalMode.MEAN_POOLING:
            return temporal.MeanPooling()
                    
        self.num_temporal_embeddings = self.positional_embedding.shape[0]
        self.transformer_width = self.clip.ln_final.weight.shape.shape[0]
        self.heads = self.transformer_width // 64
        
        assert self.num_temporal_embeddings <= max_temporal_embeddings
        
        temporal_trans = TemporalTransformer(
            width=self.transformer_width, 
            layers=self.num_temporal_hidden_layers,
            heads=self.transformer_heads, 
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