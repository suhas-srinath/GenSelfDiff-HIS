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


folder1 = './data/test_images'
folder2 = './data/test_masks'

# directories for images and labels
TRAIN_OUT_FOLDER = './train/full_size_images'
os.makedirs(TRAIN_OUT_FOLDER, exist_ok=True)
TEST_OUT_FOLDER = './test/full_size_images'
os.makedirs(TEST_OUT_FOLDER, exist_ok=True)

TRAIN_LABEL_FOLDER = './train/full_size_masks'
os.makedirs(TRAIN_LABEL_FOLDER, exist_ok=True)
TEST_LABEL_FOLDER = './test/full_size_masks'
os.makedirs(TEST_LABEL_FOLDER, exist_ok=True)

print('===========================================================================')
print('                          IMAGES and LABELS                                    ')
print('===========================================================================')

total_list1 = os.listdir(folder1)
total_list1 = np.array(sorted(total_list1, key=lambda x: int(x.split('_')[-1].split('.jpg')[0])))

total_list2 = os.listdir(folder2)
total_list2 = np.array(sorted(total_list2, key=lambda x: int(x.split('_')[-1].split('.jpg')[0])))

ratio = 0.8
idxs = np.random.RandomState(2023).permutation(total_list1.shape[0])
split = int(total_list1.shape[0] * ratio)
split_1 = idxs[:split]
split_2 = idxs[split:]
train_images, test_images = total_list1[split_1], total_list1[split_2]
train_labels, test_labels = total_list2[split_1], total_list2[split_2]

i = 0
for img_name in train_images:
    img = cv2.imread(os.path.join(folder1, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    TRAIN_OUT_IMAGE_PATH = os.path.join(TRAIN_OUT_FOLDER, 'image_' + str(i) + '.jpg')
    cv2.imwrite(TRAIN_OUT_IMAGE_PATH, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(img_name)
    i += 1

print('='*30)

i = 0
for label_name in train_labels:
    mask = cv2.imread(os.path.join(folder2, label_name))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    TRAIN_OUT_IMAGE_PATH = os.path.join(TRAIN_LABEL_FOLDER, 'label_' + str(i) + '.jpg')
    cv2.imwrite(TRAIN_OUT_IMAGE_PATH, mask)
    print(label_name)
    i += 1

print('='*30)

i = 0
for img_name in test_images:
    img = cv2.imread(os.path.join(folder1, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    TEST_OUT_IMAGE_PATH = os.path.join(TEST_OUT_FOLDER, 'image_' + str(i) + '.jpg')
    cv2.imwrite(TEST_OUT_IMAGE_PATH, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(img_name)
    i += 1

print('='*30)

i = 0
for label_name in test_labels:
    mask = cv2.imread(os.path.join(folder2, label_name))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    TEST_OUT_IMAGE_PATH = os.path.join(TEST_LABEL_FOLDER, 'label_' + str(i) + '.jpg')
    cv2.imwrite(TEST_OUT_IMAGE_PATH, mask)
    print(label_name)
    i += 1
