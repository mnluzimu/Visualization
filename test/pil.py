from PIL import Image
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np

img = Image.open(r'../data/oxbuild_images-v1/all_souls_000001.jpg')
img = img.convert('RGB')
print(img)
transform = transforms.ToTensor()
img = transform(img)
print(img.shape)
img = img.permute((1, 2, 0))[:, :, [2, 1, 0]]
img = img.numpy()
img = img * 255
img = img.astype(np.uint8)
cv.imshow('img', img)

img_cv = cv.imread(r'../data/oxbuild_images-v1/all_souls_000001.jpg')
print(img_cv.shape)

cv.imshow('img_cv', img_cv)

cv.waitKey(0)



