import os
from PIL import Image
import glob
import numpy as np
import cv2
from scipy.io import savemat

folder = './GLaS_data/*'
train_images = './data/train_images'
os.makedirs(train_images, exist_ok=True)
train_labels = './data/train_masks'
os.makedirs(train_labels, exist_ok=True)

IMGS = glob.glob(folder)
for img_path in IMGS:
    image_type = img_path.split('/')[-1].split('_')[0][:5]
    if image_type == 'train':
        is_annotation = img_path.split('/')[-1].split('_')[-1].split('.bmp')[0]
        if is_annotation != 'anno':
            img_num = int(img_path.split('/')[-1].split('_')[1].split('.bmp')[0])
            # print(image_type+str(img_num))
            print(img_num - 1)

            img = Image.open(img_path)
            img = np.array(img)
            save_path = os.path.join(train_images, 'image_' + str(img_num-1) + '.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        else:
            img_num = int(img_path.split('/')[-1].split('_')[1])
            # print(image_type+str(img_num))
            print(img_num - 1)

            mask = Image.open(img_path)
            mask = np.array(mask)
            mask[mask != 0] = 1
            save_path = os.path.join(train_labels, 'label_' + str(img_num - 1) + '.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(255 * mask, cv2.COLOR_RGB2BGR))
print('='*30)

test_images = './data/test_images'
os.makedirs(test_images, exist_ok=True)
test_labels = './data/test_masks'
os.makedirs(test_labels, exist_ok=True)

IMGS = glob.glob(folder)
indexing = 0
for img_path in IMGS:
    image_type = img_path.split('/')[-1].split('_')[0][:5]
    if image_type == 'testA':
        is_annotation = img_path.split('/')[-1].split('_')[-1].split('.bmp')[0]
        if is_annotation != 'anno':
            img_num = int(img_path.split('/')[-1].split('_')[1].split('.bmp')[0])
            # print(image_type+str(img_num))
            print(img_num - 1)

            img = Image.open(img_path)
            img = np.array(img)
            save_path = os.path.join(test_images, 'image_' + str(img_num - 1) + '.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            indexing += 1
        else:
            img_num = int(img_path.split('/')[-1].split('_')[1])
            # print(image_type + str(img_num))
            print(img_num-1)

            mask = Image.open(img_path)
            mask = np.array(mask)
            mask[mask != 0] = 1
            save_path = os.path.join(test_labels, 'label_' + str(img_num-1) + '.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(255*mask, cv2.COLOR_RGB2BGR))


ref_index = indexing-1
for img_path in IMGS:
    image_type = img_path.split('/')[-1].split('_')[0][:5]
    if image_type == 'testB':
        is_annotation = img_path.split('/')[-1].split('_')[-1].split('.bmp')[0]
        if is_annotation != 'anno':
            img_num = int(img_path.split('/')[-1].split('_')[1].split('.bmp')[0])
            # print(image_type+str(img_num))
            print(img_num + ref_index)

            img = Image.open(img_path)
            img = np.array(img)
            save_path = os.path.join(test_images, 'image_' + str(img_num + ref_index) + '.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            img_num = int(img_path.split('/')[-1].split('_')[1])
            # print(image_type + str(img_num))
            print(img_num+ref_index)

            mask = Image.open(img_path)
            mask = np.array(mask)
            mask[mask != 0] = 1
            save_path = os.path.join(test_labels, 'label_' + str(img_num+ref_index) + '.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(255*mask, cv2.COLOR_RGB2BGR))

print('='*30)

