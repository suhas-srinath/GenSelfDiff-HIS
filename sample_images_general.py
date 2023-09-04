import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from skimage import io
import cv2

folder1 = '/home/vishnupv/vishnu/AIIMS IITD IISC May 2022/unlabelled_img_patches'
folder2 = '/home/vishnupv/vishnu/MoNuSeg/unlabelled_img_patches'
folder3 = '/home/vishnupv/vishnu/GLaS/unlabelled_img_patches'


total_list1 = os.listdir(folder1)
total_list1 = sorted(total_list1, key=lambda x: int(x.split('_')[-1].split('.jpg')[0]))
path1_list = [os.path.join(folder1, f) for f in total_list1]

total_list2 = os.listdir(folder2)
total_list2 = sorted(total_list2, key=lambda x: int(x.split('_')[-1].split('.jpg')[0]))
path2_list = [os.path.join(folder2, f) for f in total_list2]

total_list3 = os.listdir(folder3)
total_list3 = sorted(total_list3, key=lambda x: int(x.split('_')[-1].split('.jpg')[0]))
path3_list = [os.path.join(folder3, f) for f in total_list3]

img_list = np.array(path1_list + path2_list + path3_list)

num_images = 1000
idxs = np.random.RandomState(2023).permutation(30149)

OUT_FOLDER = './samp_images_general'
os.makedirs(OUT_FOLDER, exist_ok=True)

for i in range(num_images):
    img = cv2.imread(img_list[idxs[i]])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    OUT_IMAGE_PATH = os.path.join(OUT_FOLDER, 'image_' + str(i) + '.jpg')
    cv2.imwrite(OUT_IMAGE_PATH, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))