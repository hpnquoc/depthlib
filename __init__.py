DEPTH_PRETRAINED_PATH = 'zoo/depth/'
MODEL_CARDS = {
    'DepthAnythingv2': {
        'vits': {
            'name' : 'depth_anything_v2_small',
            'configs': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}, 
            'path': f'{DEPTH_PRETRAINED_PATH}/depth_anything/Depth-Anything-V2-small-hf'
        },
        'vitb': {
            'name' : 'depth_anything_v2_base',
            'configs': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}, 
            'path': f'{DEPTH_PRETRAINED_PATH}/depth_anything/Depth-Anything-V2-base-hf'
        },
        'vitl': {
            'name' : 'depth_anything_v2_large',
            'configs': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}, 
            'path': f'{DEPTH_PRETRAINED_PATH}/depth_anything/Depth-Anything-V2-large-hf'
        }
    },
    'VGGT': {
        'vggt_1b': {
            'name': 'vggt_1b_depth',
            'configs': {'img_size': 518, 'patch_size': 14, 'embed_dim': 1024},
            'path': f'{DEPTH_PRETRAINED_PATH}/vggt/VGGT-1B-Depth'
        }
    }
}