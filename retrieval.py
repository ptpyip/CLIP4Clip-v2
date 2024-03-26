"""This file contains the code for training and eval on retrieval task"""


import os
import torch

from clip.clip import _transform as transform

from model import CLIP4Clip

MODELS = [
    "meanP-ViT-B/16","meanP-ViT-B/32",
    # "maxP-ViT-B/16","maxP-ViT-B/32",
    # "Trans-ViT-B/16","Trans-ViT-B/32"
]

def load(path: str, model_name="meanP-ViT-B/16", device="cpu"):
    """Load a CLIP4Clip model for inference"""
    assert model_name in MODELS
    
    if not os.path.exists(path):
        raise RuntimeError(f"Model {model_name} not found with path: {path}")
        # return None
    
    state_dict = torch.load(path, map_location="cpu")
    
    clip_name = model_name.split("-", 1)[1]
    model = CLIP4Clip(clip_name)
    model.load_state_dict(state_dict)
    model.to(device).float().eval()
    
    return model, transform(model.input_resolution)


def test_load():
    model, transform = load(
        "/csproject/dan3/downloads/ckpts/meanP-ViT-B-16.bin.3",
        device="cuda"
    )
    print(model.max_num_frame)
    
if __name__ == "__main__":
    test_load()