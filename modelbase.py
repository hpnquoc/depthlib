import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError
    
    def to(self, device):
        super(ModelBase, self).to(device)
        self.device = device
        return self

    # def load_pretrained(self, path):
    #     self.load_state_dict(torch.load(path, map_location="cpu"))

    # def save_pretrained(self, path):
    #     torch.save(self.state_dict(), path)