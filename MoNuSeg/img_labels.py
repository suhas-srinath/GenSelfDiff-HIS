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


def split_data(images, labels, ratio):
    idxs = np.random.RandomState(2023).permutation(images.shape[0])
    split = int(images.shape[0] * ratio)
    split_1 = idxs[:split]
    split_2 = idxs[split:]
    return images[split_1], images[split_2], labels[split_1], labels[split_2]


folder1 = './data/test_images/*'
# folder2 = './data/test_masks/*'

# directories for images and labels
TRAIN_OUT_FOLDER = './train/full_size_images'
os.makedirs(TRAIN_OUT_FOLDER, exist_ok=True)
TEST_OUT_FOLDER = './test/full_size_images'
os.makedirs(TEST_OUT_FOLDER, exist_ok=True)

print('===========================================================================')
print('                          IMAGES and LABELS                                    ')
print('===========================================================================')

IMAGES = []
IMGS = glob.glob(folder1)
for img_path in sorted(IMGS, key=lambda x: int(x.split("_")[-1].split(".jpg")[0])):
    print(img_path.split("_")[-1].split(".jpg")[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    IMAGES.append(img)
IMAGES = np.array(IMAGES)
print(f'number of images: {IMAGES.shape[0]}')

label_file = loadmat('./data/test_masks.mat')
all_labels = label_file['data']
LABELS = []
for i in range(all_labels.shape[0]):
    mask = all_labels[i]
    print(np.unique(mask))
    LABELS.append(mask)
LABELS = np.array(LABELS)
print(f'number of images: {LABELS.shape[0]}')

train_imgs, test_imgs, train_lbls, test_lbls = split_data(IMAGES, LABELS, 0.8)


for i in range(train_imgs.shape[0]):
    TRAIN_OUT_IMAGE_PATH = os.path.join(TRAIN_OUT_FOLDER, 'image_' + str(i) + '.jpg')
    cv2.imwrite(TRAIN_OUT_IMAGE_PATH, cv2.cvtColor(train_imgs[i], cv2.COLOR_RGB2BGR))

print(f'Number of training images: {train_imgs.shape[0]}')

path = os.path.join('./train', 'full_size_labels.mat')
mdic = {"data": train_lbls, "label": "train_labels"}
savemat(path, mdic)
print(f'Number of training labels: {train_lbls.shape[0]}')

for i in range(test_imgs.shape[0]):
    TEST_OUT_IMAGE_PATH = os.path.join(TEST_OUT_FOLDER, 'image_' + str(i) + '.jpg')
    cv2.imwrite(TEST_OUT_IMAGE_PATH, cv2.cvtColor(test_imgs[i], cv2.COLOR_RGB2BGR))

print(f'Number of testing images: {test_imgs.shape[0]}')

path = os.path.join('./test', 'full_size_labels.mat')
mdic = {"data": test_lbls, "label": "test_labels"}
savemat(path, mdic)
print(f'Number of testing labels: {test_lbls.shape[0]}')

