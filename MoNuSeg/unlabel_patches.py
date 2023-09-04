from PIL import Image
from torchvision.transforms import transforms
import torch
import glob
import json
import numpy as np
import os
import cv2
from skimage import io
import sys

folder = './data/train_images/*'

# directories for images
OUT_FOLDER = './unlabelled_img_patches'
os.makedirs(OUT_FOLDER, exist_ok=True)

PATCHES = []
indexing = 0
IMGS = glob.glob(folder)
for img_path in sorted(IMGS, key=lambda x: int(x.split("_")[-1].split('.jpg')[0])):
    image_number = int(img_path.split("_")[-1].split(".jpg")[0])

    img = cv2.imread(img_path)
    ximg = transforms.ToTensor()(img)

    size = 256  # patch size
    stride = 64  # patch stride
    patches = ximg.unfold(1, size, stride).unfold(2, size, stride)
    patches = patches.reshape(img.shape[2], -1, size, size)
    patches = torch.permute(patches, (1, 2, 3, 0))
    patches = patches.numpy()

    for i in range(patches.shape[0]):
        save_path = os.path.join(OUT_FOLDER, 'image_' + str(indexing) + '.jpg')
        cv2.imwrite(save_path, np.uint8(255*patches[i, :, :, :]))
        indexing += 1
    print('{} patches are created'.format(indexing))
