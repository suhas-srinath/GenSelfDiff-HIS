# This script is used to obtain the test image-label patches form the full-scale image-labels

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
from scipy.io import loadmat, savemat


folder1 = './test/full_size_images/*'
folder2 = './test/full_size_masks/*'

# directories for images and labels
OUT_FOLDER = './test/images'
os.makedirs(OUT_FOLDER, exist_ok=True)

print('===========================================================================')
print('                          IMAGE PATCHES                                    ')
print('===========================================================================')
indexing = 0
IMGS = glob.glob(folder1)
for img_path in sorted(IMGS, key=lambda x: int(x.split("_")[-1].split(".jpg")[0])):
    print(img_path.split("_")[-1].split(".jpg")[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ximg = transforms.ToTensor()(img)

    size = 256  # patch size
    stride = 64  # patch stride
    patches = ximg.unfold(1, size, stride).unfold(2, size, stride)
    patches = patches.reshape(img.shape[2], -1, size, size)
    patches = torch.permute(patches, (1, 2, 3, 0))
    patches = patches.numpy()

    for i in range(patches.shape[0]):
        save_path = os.path.join(OUT_FOLDER, 'image_' + str(indexing) + '.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(np.uint8(255 * patches[i, :, :, :]), cv2.COLOR_RGB2BGR))
        indexing += 1
print('Number of image patches in total: {}'.format(indexing))

print('===========================================================================')
print('                          LABEL PATCHES                                    ')
print('===========================================================================')

LABEL_PATCHES = []
indexing = 0
IMGS = glob.glob(folder2)
for label_path in sorted(IMGS, key=lambda x: int(x.split("_")[-1].split(".jpg")[0])):
    print(label_path.split("_")[-1].split(".jpg")[0])
    mask = cv2.imread(label_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    xlabel = transforms.ToTensor()(mask)

    size = 256  # patch size
    stride = 64  # patch stride
    patches = xlabel.unfold(1, size, stride).unfold(2, size, stride)
    patches = patches.reshape(-1, size, size)
    patches = np.uint8(patches.numpy())

    for i in range(patches.shape[0]):
        LABEL_PATCHES.append(patches[i, :, :])
        indexing += 1
LABEL_PATCHES = np.array(LABEL_PATCHES)
print('Number of label patches in total: {}'.format(LABEL_PATCHES.shape[0]))
print(np.unique(LABEL_PATCHES))

print('===========================================================================')
print('                            Now Saving the patches                         ')
print('===========================================================================')


path = os.path.join('./test', 'label_patches.mat')
mdic = {"data": LABEL_PATCHES, "label": "test_labels"}
savemat(path, mdic)
print(f'Number of training labels: {LABEL_PATCHES.shape[0]}')
