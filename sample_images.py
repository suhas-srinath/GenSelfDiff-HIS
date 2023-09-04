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


folder1 = '../../unlabelled_img_patches/'
num_images = 1000
idxs = np.random.RandomState(2023).permutation(5328)

OUT_FOLDER = './samp_images'
os.makedirs(OUT_FOLDER, exist_ok=True)

for i in range(num_images):
    img_path = folder1 + 'image_' + str(idxs[i]) + '.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    OUT_IMAGE_PATH = os.path.join(OUT_FOLDER, 'image_' + str(i) + '.jpg')
    cv2.imwrite(OUT_IMAGE_PATH, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))