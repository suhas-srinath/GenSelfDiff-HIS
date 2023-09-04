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


# def split_data(images, labels, ratio):
#     idxs = np.random.RandomState(2022).permutation(images.shape[0])
#     split = int(images.shape[0] * ratio)
#     split_1 = idxs[:split]
#     split_2 = idxs[split:]
#     return images[split_1], images[split_2], labels[split_1], labels[split_2]


folder1 = './data/labelled_images/*'

# directories for images and labels
TRAIN_OUT_FOLDER = './train/images'
os.makedirs(TRAIN_OUT_FOLDER, exist_ok=True)
TEST_OUT_FOLDER = './test/images'
os.makedirs(TEST_OUT_FOLDER, exist_ok=True)


print('===========================================================================')
print('                          IMAGE PATCHES                                    ')
print('===========================================================================')
IMG_PATCHES = []
indexing = 0
IMGS = glob.glob(folder1)
for img_path in sorted(IMGS, key=lambda x: int(x.split("_")[-1].split(".jpg")[0])):
    print(img_path.split("_")[-1].split(".jpg")[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ximg = transforms.ToTensor()(img)

    size = 256  # patch size
    stride = 256  # patch stride
    patches = ximg.unfold(1, size, stride).unfold(2, size, stride)
    patches = patches.reshape(img.shape[2], -1, size, size)
    patches = torch.permute(patches, (1, 2, 3, 0))
    patches = patches.numpy()

    for i in range(patches.shape[0]):
        IMG_PATCHES.append(np.uint8(255 * patches[i, :, :, :]))
        indexing += 1
IMG_PATCHES = np.array(IMG_PATCHES)
print('Number of image patches in total: {}'.format(IMG_PATCHES.shape[0]))

print('===========================================================================')
print('                          LABEL PATCHES                                    ')
print('===========================================================================')
LABEL_PATCHES = []
label_file = loadmat('./data/masks.mat')
all_labels = label_file['data']
indexing = 0
for i in range(all_labels.shape[0]):
    xlabel = transforms.ToTensor()(all_labels[i, :, :])

    size = 256  # patch size
    stride = 256  # patch stride
    patches = xlabel.unfold(1, size, stride).unfold(2, size, stride)
    patches = patches.reshape(-1, size, size)
    patches = np.uint8(255 * patches.numpy())

    for i in range(patches.shape[0]):
        LABEL_PATCHES.append(patches[i, :, :])
        indexing += 1
    print('{} label patches are created'.format(indexing))
LABEL_PATCHES = np.array(LABEL_PATCHES)
print('Number of label patches in total: {}'.format(LABEL_PATCHES.shape[0]))

print('===========================================================================')
print('             Now Saving the patches into train and test folders            ')
print('===========================================================================')

# train_img_patches, test_img_patches, train_lbl_patches, test_lbl_patches = split_data(IMG_PATCHES, LABEL_PATCHES, 0.8)

ratio = 0.8
total_index = [i for i in range(IMG_PATCHES.shape[0])]
train_index = total_index[:int(ratio * len(total_index))]
test_index = total_index[int(ratio * len(total_index)):]
train_img_patches, train_lbl_patches = IMG_PATCHES[train_index], LABEL_PATCHES[train_index]
test_img_patches, test_lbl_patches = IMG_PATCHES[test_index], LABEL_PATCHES[test_index]

for i in range(train_img_patches.shape[0]):
    TRAIN_OUT_IMAGE_PATH = os.path.join(TRAIN_OUT_FOLDER, 'image_' + str(i) + '.jpg')
    cv2.imwrite(TRAIN_OUT_IMAGE_PATH, cv2.cvtColor(train_img_patches[i, :, :, :], cv2.COLOR_RGB2BGR))

print(f'Number of training images: {train_img_patches.shape[0]}')

path = os.path.join('./train', 'label_patches.mat')
mdic = {"data": train_lbl_patches, "label": "train_labels"}
savemat(path, mdic)
print(f'Number of training labels: {train_lbl_patches.shape[0]}')

for i in range(test_img_patches.shape[0]):
    TEST_OUT_IMAGE_PATH = os.path.join(TEST_OUT_FOLDER, 'image_' + str(i) + '.jpg')
    cv2.imwrite(TEST_OUT_IMAGE_PATH, cv2.cvtColor(test_img_patches[i, :, :, :], cv2.COLOR_RGB2BGR))

print(f'Number of testing images: {test_img_patches.shape[0]}')

path = os.path.join('./test', 'label_patches.mat')
mdic = {"data": test_lbl_patches, "label": "test_labels"}
savemat(path, mdic)
print(f'Number of testing labels: {test_lbl_patches.shape[0]}')

