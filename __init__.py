import importlib
from .configs import load_config
import torch

import os
import sys

sys.path.append(os.path.dirname(__file__))

DEPTH_PRETRAINED_PATH = 'zoo/depth/'
MODEL_CARDS = {
    'depthanythingv2': {
        'module': 'depth_anything_v2.dpt',
        'model': 'DepthAnythingV2',
        'vits': {
            'name' : 'depth_anything_v2_small',
            'configs': 'depth_anything_v2/vits_pretrain',
            'path': f'{DEPTH_PRETRAINED_PATH}/depth_anything/Depth-Anything-V2-small-hf/depth_anything_v2_vits.pth',
            'url': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth'
        },
        'vitb': {
            'name' : 'depth_anything_v2_base',
            'configs': 'depth_anything_v2/vitb_pretrain', 
            'path': f'{DEPTH_PRETRAINED_PATH}/depth_anything/Depth-Anything-V2-base-hf/depth_anything_v2_vitb.pth',
            'url': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth'
        },
        'vitl': {
            'name' : 'depth_anything_v2_large',
            'configs': 'depth_anything_v2/vitl_pretrain', 
            'path': f'{DEPTH_PRETRAINED_PATH}/depth_anything/Depth-Anything-V2-large-hf/depth_anything_v2_vitl.pth',
            'url': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth'
        }
    },
    'vggt': {
        'module': 'vggt.vggt_depth',
        'model': 'VGGT',
        'vggt_1b': {
            'name': 'vggt_1b_depth',
            'configs': 'vggt/1b_pretrain',
            'path': f'{DEPTH_PRETRAINED_PATH}/vggt/VGGT-1B-Depth/model.pt',
            'url': 'https://huggingface.co/hpnquoc/VGGT-1B-Depth/resolve/main/model.pt'
        }
    }
}

def get_model(model_name, variant):
    """
    Retrieve a model based on the provided model name and variant.
    
    Args:
        model_name (str): The name of the model to retrieve.
        variant (str): The specific variant of the model.
    
    Returns:
        nn.Module: The instantiated model with the specified configuration and pre-trained weights.
    
    Raises:
        ValueError: If the model name or variant is not found in the predefined model cards.
    """
    model_name = model_name.lower()
    variant = variant.lower()
    # check if model_name exists
    if model_name not in MODEL_CARDS:
        raise ValueError(f"Model name {model_name} not found in available models {MODEL_CARDS.keys()}")
    
    # check if variant exists
    if variant not in MODEL_CARDS[model_name]:
        raise ValueError(f"Variant {variant} not found in available variants {[i for i in MODEL_CARDS[model_name].keys() if i != 'module' and i != 'model']}")
    
    model_conf = MODEL_CARDS[model_name][variant]
    module = importlib.import_module(MODEL_CARDS[model_name]['module'])
    model = getattr(module, MODEL_CARDS[model_name]['model'])(**load_config(model_conf['configs'])['model'])
    model.load_state_dict(torch.load(model_conf['path']))
    return model