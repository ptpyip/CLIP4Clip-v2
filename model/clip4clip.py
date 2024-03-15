import os
from enum import Enum

import torch
from torch import nn

from clip.model import CLIP, convert_weights

CLIP_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
}

class TemporalMode(Enum):
    MEAN_POOLING = 0
    TRANSFORMER = 1
    # CrossTransformer = 2        # not used.

def build_model(state_dict: dict):
    """ build model from a given state dict
    if state_dict is {} => build from empty
    """
    
class CLIP4Clip(nn.Module):
    CLIP_NAME = "ViT-B/32"
    
    def __init__(self, clip_name=CLIP_NAME):
        self.clip = self._init_clip(clip_name)
        
    def _init_clip(self, model_name, device="cpu"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViT-B-32.pt")
        
        if model_name in CLIP_NAME:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), CLIP_NAME[model_name]
            )

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model {model_name} not found; available models = {list(CLIP_NAME.keys())}"
            )

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
            
           
        return self.build_clip(state_dict)

    # model = build_clip(state_dict)
    # model.train(True)
    
    # return model
        
    def build_clip(self, state_dict: dict):
        """Copied form the clip package"""
        vit = "visual.proj" in state_dict

        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

        model = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        convert_weights(model)
        
        return model