import glob
import json
import numpy as np
import os
import cv2
from skimage import io
from skimage.draw import disk


Annotated_Images_folder = './annotated images/*'

# directories for images and labels
OUT_IMAGES_FOLDER = './data/labelled_images'
os.makedirs(OUT_IMAGES_FOLDER,exist_ok=True)
OUT_LABELS_FOLDER = './data/labels'
os.makedirs(OUT_LABELS_FOLDER,exist_ok=True)

LABELS = {'Malignant':1, 'Non malignant stroma':2, 'Non malignant epithelium':3}
colors = [(255,0,0), (0,255,0), (0,0,255)]

print('=====================================================')

indexing = 0
length = 0
len_list = []
all_masks_list = []
for item in glob.glob(Annotated_Images_folder):
    Images_with_Annotations = []
    print(item)

    Annotated_Images = [image_path for image_path in sorted(glob.glob(item + "/*.jpg"))]
    Annotated_Images_paths = [image_path.split("/")[-1] for image_path in Annotated_Images]

    Annotations_files = [Annotations_file for Annotations_file in sorted(glob.glob(item + "/*.json"))]
    Annotations_paths = []
    cnt = 0
    Annotations = []
    for Annotations_file in Annotations_files:
        Annotations = json.load(open(Annotations_file))
        path_temp = []

        for i in range(len(Annotations)):
            if (Annotations[i]['Label'] != 'Skip'):
                path_temp.append(Annotations[i]['External ID'])

        Annotations_paths.append(sorted(path_temp))

    # if (len(Annotations_paths) > 1):
    #     print('Two annoations in the folder are equal or not ?: {} '.format(Annotations_paths[0] == Annotations_paths[1]))

    if (len(Annotated_Images_paths) > len(Annotations_paths[0])):
        [Images_with_Annotations.append(i) for i in Annotated_Images_paths if i in Annotations_paths[0]]
    else:
        [Images_with_Annotations.append(i) for i in Annotations_paths[0] if i in Annotated_Images_paths]

    Images_with_Annotations = list(set(Images_with_Annotations))
    Images_with_Annotations.sort()
    for image_path in Images_with_Annotations:

        IMAGE = io.imread(os.path.join(item, image_path))
        IMG_SIZE = IMAGE.shape
        print(indexing, image_path)

        OUT_IMAGE_PATH = os.path.join(OUT_IMAGES_FOLDER, 'image_' + str(indexing) + '.jpg')
        OUT_LABEL_PATH = os.path.join(OUT_LABELS_FOLDER, 'label_' + str(indexing) + '.jpg')

        for i in range(len(Annotations)):
            if image_path in Annotations[i]['External ID']:
                Cancer_Cells = dict(Annotations[i]['Label'])
                len_list.append(indexing)
                break

        image_label = np.zeros((IMG_SIZE[0],IMG_SIZE[1],3),dtype='uint8')

        for cancer_cell in Cancer_Cells.keys():

            if(cancer_cell[-1] == ' '):
                cancer_cell_key = cancer_cell[:-1]
            else:
                cancer_cell_key = cancer_cell

            contours = []
            for i in range(len(Cancer_Cells[cancer_cell])):
                temp_list = []
                for positions in (Cancer_Cells[cancer_cell][i]['geometry']):
                    row = positions['y'] if positions['y'] == 0 else positions['y'] - 1
                    column = positions['x'] if positions['x'] == 0 else positions['x'] - 1
                    temp_list.append([[column, row]])
                contours.append(np.array(temp_list))
            # cv2.drawContours(image_label, contours, -1, colors[LABELS[cancer_cell_key]-1], thickness=cv2.FILLED) # for color segmentation map
            cv2.drawContours(image_label, contours, -1, LABELS[cancer_cell_key], thickness=cv2.FILLED)

        io.imsave(OUT_IMAGE_PATH, IMAGE)
        io.imsave(OUT_LABEL_PATH, image_label)
        all_masks_list.append(image_label)

        indexing = indexing + 1
        # print('----------------------------------------------------')

    length =length+len(Images_with_Annotations)
    # print('{} image-label pairs are created'.format(length))
print(len(len_list))

masks = np.array(all_masks_list)
path = os.path.join('./data', 'masks.mat')
mdic = {"data": masks, "label": "labels"}
savemat(path, mdic)
