# Depth Estimation Library

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue) ![PyTorch](https://img.shields.io/badge/pytorch-1.9%2B-red)

This repository contains a lightweight and modular **Depth Estimation Library** designed for easy integration into various computer vision pipelines. The library supports multiple state-of-the-art models for depth estimation.

## Table of Contents

- [Depth Estimation Library](#depth-estimation-library)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Quick Start](#quick-start)
  - [Model Cards](#model-cards)
    - [DepthAnythingV2](#depthanythingv2)
    - [VGGT](#vggt)
  - [Usage Examples](#usage-examples)
  - [Pre-trained Models](#pre-trained-models)
  - [Contributing](#contributing)
  - [Citation](#citation)
  - [License](#license)

---

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.9 or higher
- CUDA (optional, for GPU acceleration)
- OpenCV (`pip install opencv-python`)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/hpnquoc/depthlib.git
   cd depthlib
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained models (see [Pre-trained Models](#pre-trained-models)) and place them in the appropriate directory.

---

## Quick Start

Here’s how to quickly use the library to estimate depth from an image:

```python
from lib.depthlib import get_model
import cv2

# Load the model (replace '<model_name>' and '<variant>' with your choice)
model = get_model('vggt', 'vggt_1b').to('cuda').eval()

# Load an image
img = cv2.imread('image.jpg')

# Perform inference
depth = model.infer_image(img)

# Post-process the depth map for visualization
depth = depth.cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to [0, 1]
depth = (depth * 255).astype('uint8')  # Scale to [0, 255]

# Display the depth map
cv2.imshow('Depth Map', depth)
cv2.waitKey(0)
```

---

## Model Cards

The library supports the following models and variants:

### DepthAnythingV2

| Variant | Name                     | Configs                  | Pre-trained Path                                                                 |
|---------|--------------------------|--------------------------|----------------------------------------------------------------------------------|
| Small   | `depth_anything_v2_small` | `depth_anything_v2/vits_pretrain` | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Base    | `depth_anything_v2_base`  | `depth_anything_v2/vitb_pretrain` | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Large   | `depth_anything_v2_large` | `depth_anything_v2/vitl_pretrain` | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |

### VGGT

| Variant | Name           | Configs            | Pre-trained Path                                                |
|---------|----------------|--------------------|------------------------------------------------------------------|
| 1B      | `vggt_1b_depth` | `vggt/1b_pretrain` | [Download](https://huggingface.co/hpnquoc/VGGT-1B-Depth/resolve/main/model.pt?download=true)  |

---

## Usage Examples

```python
from lib.depthlib import get_model
import cv2

# Load the model
model = get_model('<model_name>', '<variant>').to('cuda').eval()

# Load and preprocess the image
img = cv2.imread('example.jpg')

# Infer depth
depth = model.infer_image(img)

# Normalize and visualize the depth map
depth = depth.cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min())
depth = (depth * 255).astype('uint8')
cv2.imshow('Depth Map', depth)
cv2.waitKey(0)
```

## Pre-trained Models

Pre-trained models can be downloaded from the following links:

- **DepthAnythingV2**: [Download Link](#)
- **VGGT**: [Download Link](#)

Place the downloaded weights in the `pretrained/` directory and ensure the paths in `MODEL_CARDS` are correctly configured.

---

## Contributing

We welcome contributions to improve this project! If you find any issues or have suggestions, please open an issue or submit a pull request. For major changes, please discuss them in the issue tracker first.

---

## Citation

If you use this library in your research, please cite it as follows:

```bibtex
@misc{depthlib,
  author = {Ho Pham Nam Quoc},
  title = {DepthLib},
  year = {2025},
  howpublished = {\url{https://github.com/hpnquoc/depthlib}}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
