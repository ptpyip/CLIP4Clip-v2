"""This file contains the code for training and eval on retrieval task"""

import os
import yaml
import torch

from clip.clip import _transform as transform

from model import CLIP4Clip

MODELS = [
    "meanP-ViT-B/16","meanP-ViT-B/32",
    # "maxP-ViT-B/16","maxP-ViT-B/32",
    # "Trans-ViT-B/16","Trans-ViT-B/32"
]

def init_model(path: str, model_name="meanP-ViT-B/16", device="cpu"):
    """Load a CLIP4Clip model for inference"""
    assert model_name in MODELS
    
    if not os.path.exists(path):
        raise RuntimeError(f"Model {model_name} not found with path: {path}")
        # return None
    
    state_dict = torch.load(path, map_location="cpu")
    
    clip_name = model_name.split("-", 1)[1]
    model = CLIP4Clip(clip_name)
    # model.load_state_dict(state_dict)
    model.to(device).float()
    
    return model, transform(model.input_resolution)

# def pre_process(self, moments):
#         assert isinstance(moments, list)

#         bs = len(moments)
#         L = self.model.max_num_frame
#         H = W = self.model.input_resolution

#         moment_tensor = torch.zeros(bs, L, 3, H, W)
#         moment_mask = torch.zeros(bs, L)
#         for i, moment in enumerate(moments):
#             moment_length = len(moment)

#             if moment_length > L:
#                 print(f"Warning moment too large: {moment_length}, slicing is used")

#                 ## hard slice
#                 moment = moment[:L]
#                 moment_length = L

#             moment_tensor[i][:moment_length] = torch.stack(
#                 [self.transform(frame) for frame in moment]
#             )
#             moment_mask[i][:moment_length] += 1

#         return moment_tensor, moment_mask


def test_load():
    model, transform = init_model(
        "/csproject/dan3/downloads/ckpts/meanP-ViT-B-16.bin.3",
        device="cuda"
    )
    print(model.max_num_frame)
    
if __name__ == "__main__":
    # test_load()
    with open("./config/meanP-ViT-B-16-0326.yaml", 'r') as f:
        foo = yaml.safe_load(f)
        print(foo)
        
  