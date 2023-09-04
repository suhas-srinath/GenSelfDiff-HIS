import xml.etree.ElementTree as ET
import numpy as np
import skimage.draw
import os
import glob
import cv2
from PIL import Image
from scipy.io import savemat


def binary_mask_from_xml_file(xml_file_path, image_shape=(1000, 1000)):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    def vertex_element_to_tuple(vertex_element):
        col = float(vertex_element.get('X'))
        row = float(vertex_element.get('Y'))
        return round(row), round(col)

    mask = np.zeros(image_shape, dtype=np.uint8)
    for region in root.iter('Region'):
        vertices = map(vertex_element_to_tuple, region.iter('Vertex'))
        rows, cols = np.array(list(zip(*vertices)))

        rr, cc = skimage.draw.polygon(rows, cols, mask.shape)
        mask[rr, cc] = 1

    return mask

# TRAIN SET


folder1 = './MoNuSeg_train/Annotations/*'
folder2 = './MoNuSeg_train/Tissue Images'

labelled_images = './data/train_images'
os.makedirs(labelled_images, exist_ok=True)
labels = './data/train_masks'
os.makedirs(labels, exist_ok=True)

labels_list = []
IMGS = glob.glob(folder1)
indexing = 0
for mask_path in IMGS:
    img_path = os.path.join(folder2, mask_path.split('/')[-1].split('.xml')[0] + '.tif')

    image = Image.open(img_path)
    image = np.array(image)
    save_path1 = os.path.join(labelled_images, 'image_' + str(indexing) + '.jpg')
    cv2.imwrite(save_path1, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    image_label = binary_mask_from_xml_file(mask_path)
    save_path2 = os.path.join(labels, 'mask_' + str(indexing) + '.jpg')
    cv2.imwrite(save_path2, 255 * image_label)
    labels_list.append(image_label)

    indexing += 1
    if indexing % 5 == 0:
        print(f'{indexing} number of image_label pairs saved.')

masks = np.array(labels_list)
path = os.path.join('./data', 'train_masks.mat')
mdic = {"data": masks, "label": "labels"}
savemat(path, mdic)

# TEST SET

folder1 = './MoNuSeg_test/Annotations/*'
folder2 = './MoNuSeg_test/Tissue Images'

labelled_images = './data/test_images'
os.makedirs(labelled_images, exist_ok=True)
labels = './data/test_masks'
os.makedirs(labels, exist_ok=True)

labels_list = []
IMGS = glob.glob(folder1)
indexing = 0
for mask_path in IMGS:
    img_path = os.path.join(folder2, mask_path.split('/')[-1].split('.xml')[0] + '.tif')

    image = Image.open(img_path)
    image = np.array(image)
    save_path1 = os.path.join(labelled_images, 'image_' + str(indexing) + '.jpg')
    cv2.imwrite(save_path1, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    image_label = binary_mask_from_xml_file(mask_path)
    save_path2 = os.path.join(labels, 'mask_' + str(indexing) + '.jpg')
    cv2.imwrite(save_path2, 255 * image_label)
    labels_list.append(image_label)

    indexing += 1
    if indexing % 5 == 0:
        print(f'{indexing} number of image_label pairs saved.')

masks = np.array(labels_list)
path = os.path.join('./data', 'test_masks.mat')
mdic = {"data": masks, "label": "labels"}
savemat(path, mdic)

