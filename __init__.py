import importlib
from .configs import load_config
import torch

from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Normalize,
    ToTensor,
)

try:
    from utils.transform import Resize
except:
    import torchvision.transforms.functional as F
    class Resize(object):
        def __init__(self, size, resize_method = 'lower_bound', keep_aspect_ratio = True):
            super().__init__()
            self.size = size
            self.keep_aspect_ratio = keep_aspect_ratio
            self.resize_method = resize_method

        def get_size(self, width, height):
            if not self.keep_aspect_ratio:
                return (self.size, self.size)
            if self.resize_method == 'upper_bound':
                new_width = self.size
                new_height = int(height * (self.size / width))
            elif self.resize_method == 'lower_bound':
                new_height = self.size
                new_width = int(width * (self.size / height))
            else:
                raise ValueError(f"resize_method {self.resize_method} not implemented")

            return (new_height, new_width)
        
        def __call__(self, img):
            if not isinstance(img, torch.Tensor):
                width, height = img.size
            else:
                height, width = img.shape[-2:]
            size = self.get_size(width, height)
            return F.resize(img, size)

import os
import sys

sys.path.append(os.path.dirname(__file__))

DEPTH_PRETRAINED_PATH = 'zoo/depth/'
MODEL_CARDS = {
    'depthanythingv2': {
        'module': 'depth_anything_v2',
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
        'module': 'vggt',
        'model': 'VGGT',
        'vggt_1b': {
            'name': 'vggt_1b_depth',
            'configs': 'vggt/vggt_1b_pretrain',
            'path': f'{DEPTH_PRETRAINED_PATH}/vggt/VGGT-1B-Depth/model.pt',
            'url': 'https://huggingface.co/hpnquoc/VGGT-1B-Depth/resolve/main/model.pt'
        }
    },
    'depthpro': {
        'module': 'depth_pro',
        'model': 'DepthPro',
        'depthpro': {
            'name': 'depth_pro',
            'configs': 'depth_pro/depth_pro_pretrain',
            'path': f'{DEPTH_PRETRAINED_PATH}/depth_pro/depth_pro.pt',
            'url': 'https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt'
        }
    }
}

def image_transform(conf):
    list_transforms = []
    for key in conf.keys():
        if key == 'ToTensor':
            list_transforms.append(ToTensor())
        if key == 'Normalize':
            list_transforms.append(Normalize(**conf[key]))
        if key == 'ConvertImageDtype':
            list_transforms.append(ConvertImageDtype(torch.float32))
        if key == 'Resize':
            if Resize.__module__ == 'utils.transform':
                list_transforms.append(Resize(**conf[key]))
            else:
                list_transforms.append(Resize(size=conf[key]['width'], resize_method=conf[key]['resize_method'], keep_aspect_ratio=conf[key]['keep_aspect_ratio']))

    return Compose(list_transforms)

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
    model.load_state_dict(torch.load(model_conf['path']), strict=False)
    transform = image_transform(load_config(model_conf['configs'])['transform'])
    return model, transform