from torchvision import models
import torch


net = models.resnet18()
layers = list(net.children())
print(layers)
input = torch.ones((2, 3, 900, 900)).float()
print(input.dtype)
o = layers[0](input)
print(o.shape)
o = layers[1](o)
print(o.shape)
o = layers[2](o)
print(o.shape)
o = layers[3](o)
print(o.shape)
o = layers[4](o)
print(o.shape)
o = layers[5](o)
print(o.shape)
o = layers[6](o)
print(o.shape)
o = layers[7](o)
print(o.shape)
o = layers[8](o)
print(o.shape)
o = layers[9](o)
print(o.shape)


