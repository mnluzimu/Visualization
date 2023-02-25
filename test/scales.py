import numpy as np
import torch.nn as nn
import torch


input = [torch.ones((1, 3, 900, 900)).float(), 1, 2]

scales = np.logspace(0, 6, 7, base=2**(1/2)) / 4
for scale in scales:
    input_t = nn.functional.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
    print(input_t.shape)