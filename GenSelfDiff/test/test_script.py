import os
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet50
import torchvision.transforms as transforms
from torch.nn import functional as F
from scipy.io import loadmat
import cv2
import shutil
from matplotlib import pyplot as plt
# from simple_colors import *

from tqdm import tqdm
from metrics import Aggregated_jaccard_index, Hausdorff_distance, Jaccard_score, precision, sensitivity, accuracy, F1_score, conf_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import sys

sys.path.append('../downstream_train')
from model import SegNet


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Hyper Parameters
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMG_CHANNELS = 3
BATCH_SIZE = 1
NUM_CLASSES = 4
TEST_IMAGE_DIR = '../../test/images'
TEST_LABEL_PATH = '../../test/label_patches.mat'

start = "\033[1m"
end = "\033[0;0m"

# transformations to be performed on the data points
transformations = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def class_weights(target_labl):
    _, target_label1 = torch.max(target_labl, dim=0)
    weights = np.ones(NUM_CLASSES)
    target_label = target_label1.reshape(-1)
    all_labels = target_label.cpu().numpy()
    labels, label_counts = np.unique(np.array(all_labels), return_counts=True)
    w = 1 - np.round(label_counts / np.sum(label_counts), 4)
    if len(labels) == 1:
        w = 1.0
    weights[labels] = w

    print(labels)
    print(label_counts)

    return weights


def get_images_list(path1, k=None):
    total_list1 = os.listdir(path1)
    total_list1 = sorted(total_list1, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if k is None:
        return np.array(total_list1)
    else:
        return np.array(total_list1[:k])


class Histo_Dataset(Dataset):
    def __init__(self, image1_dir, image1_list, label_list, transform=None):
        self.image1_dir = image1_dir
        self.image1_list = image1_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        img1_path = os.path.join(self.image1_dir, self.image1_list[index])

        image1 = io.imread(img1_path)
        mask = self.label_list[index]

        if self.transform is not None:
            image1 = self.transform(image1)

        return image1, mask


def eval_epoch(test_loader, model, num_classes, device):
    model_type = 'diffusion'
    loss_type = 'SSFL_A'
    with torch.no_grad():
        model.eval()
        aji_scores = 0.0
        jaccard_scores = np.zeros(num_classes-1)
        fscores = np.zeros(num_classes-1)
        hds = np.zeros(num_classes-1)
        accuracies = np.zeros(num_classes-1)
        senstivities = np.zeros(num_classes-1)
        precisions = np.zeros(num_classes-1)

        # jaccard_scores = np.zeros(num_classes)
        # fscores = np.zeros(num_classes)
        # hds = np.zeros(num_classes)
        # accuracies = np.zeros(num_classes)
        # senstivities = np.zeros(num_classes)
        # precisions = np.zeros(num_classes)

        cm = np.zeros((num_classes, num_classes))
        total_items = 0
        p_bar = tqdm(test_loader)

        save_path1 = './true_labels'
        save_path2 = './predicted_labels'
        save_path3 = './images'

        if os.path.exists(save_path1):
            shutil.rmtree(save_path1)
        os.makedirs(save_path1, exist_ok=True)

        if os.path.exists(save_path2):
            shutil.rmtree(save_path2)
        os.makedirs(save_path2, exist_ok=True)

        if os.path.exists(save_path3):
            shutil.rmtree(save_path3)
        os.makedirs(save_path3, exist_ok=True)

        indexing = 0
        for img, target_label in p_bar:
            img = img.to(device)
            target_label = target_label.squeeze(1).to(device)
            target_label1 = target_label.long()
            target_label = F.one_hot(target_label1, num_classes)
            target_label = torch.permute(target_label, (0, 3, 1, 2))

            t = torch.full((BATCH_SIZE,), 0, dtype=torch.long)
            t = t.to(device)

            predicted_label = model(img, t)

            aji_score = Aggregated_jaccard_index(target_label, predicted_label, device)
            jaccard_score = Jaccard_score(target_label, predicted_label, num_classes)
            fscore = F1_score(target_label, predicted_label, num_classes)
            hd = Hausdorff_distance(target_label, predicted_label, num_classes)
            acc = accuracy(target_label, predicted_label, num_classes)
            senstvty = sensitivity(target_label, predicted_label, num_classes)
            prec = precision(target_label, predicted_label, num_classes)


            print('===============================')
            print('           IMAGE_' + str(total_items))
            print('===============================')
            print('')
            print('AJI: ', np.round(aji_score, 4))
            print('Jaccard Score: ', np.round(jaccard_score, 4), ' Mean: ',
                  np.round(np.mean(jaccard_score), 4))
            print('F1 Score: ', np.round(fscore, 4), ' Mean: ', np.round(np.mean(fscore), 4))
            print('Hausdorff Distance: ', np.round(hd, 4), ' Mean: ', np.round(np.mean(hd), 4))
            print('Accuracy: ', np.round(acc, 4), ' Mean: ', np.round(np.mean(acc), 4))
            print('Sensitivity: ', np.round(senstvty, 4), ' Mean: ', np.round(np.mean(senstvty), 4))
            print('Precision: ', np.round(prec, 4), ' Mean: ', np.round(np.mean(prec), 4))
            print('')

            aji_scores += aji_score
            jaccard_scores += jaccard_score
            fscores += fscore
            hds += hd
            accuracies += acc
            senstivities += senstvty
            precisions += prec

            cm += conf_matrix(target_label, predicted_label, num_classes)

            batch = 1  # predicted_label.shape[0]
            total_items += 1  # batch

            # True label mappings
            labels_t = np.zeros((target_label1.shape[0], target_label1.shape[1], target_label1.shape[2], 3),
                                dtype=np.uint8)
            # labels_t[target_label1.cpu() == 1] = [127, 255, 0]
            labels_t[target_label1.cpu() == 1] = [255, 69, 0]
            labels_t[target_label1.cpu() == 2] = [127, 255, 0]
            labels_t[target_label1.cpu() == 3] = [135, 206, 250]

            # Predicted label mappings
            _, pred_labels = torch.max(predicted_label, 1)
            labels_p = np.zeros((pred_labels.shape[0], pred_labels.shape[1], pred_labels.shape[2], 3),
                                dtype=np.uint8)
            # labels_p[pred_labels.cpu() == 1] = [127, 255, 0]
            labels_p[pred_labels.cpu() == 1] = [255, 69, 0]
            labels_p[pred_labels.cpu() == 2] = [127, 255, 0]
            labels_p[pred_labels.cpu() == 3] = [135, 206, 250]

            for i in range(target_label.shape[0]):
                image_label_t = labels_t[i]
                image_label_p = labels_p[i]
                out_label_path1 = os.path.join(save_path1, 'label_' + str(indexing) + '_' +
                                               str(np.round(np.mean(senstvty) / batch, 2)) + '_' +
                                               str(np.round(np.mean(prec) / batch, 2)) + '.jpg')
                cv2.imwrite(out_label_path1, cv2.cvtColor(image_label_t, cv2.COLOR_RGB2BGR))

                out_label_path2 = os.path.join(save_path2, 'label_' + str(indexing) + '_' +
                                               str(np.round(np.mean(senstvty) / batch, 2)) + '_' +
                                               str(np.round(np.mean(prec) / batch, 2)) + '.jpg')
                cv2.imwrite(out_label_path2, cv2.cvtColor(image_label_p, cv2.COLOR_RGB2BGR))

                image = torch.permute(img[i], (1, 2, 0))
                image = image.cpu().numpy()
                image = np.uint8(255 * image)
                out_image_path = os.path.join(save_path3, 'image_' + str(indexing) + '_' +
                                              str(np.round(np.mean(senstvty) / batch, 2)) + '_' +
                                              str(np.round(np.mean(prec) / batch, 2)) + '.jpg')
                cv2.imwrite(out_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                wt1 = class_weights(target_label[i])
                wt2 = class_weights(predicted_label[i])
                temp_t = target_label[i] * predicted_label[i]
                # temp_t[temp_t == 0] = 0.5
                for cls in range(NUM_CLASSES):
                    print([torch.min(temp_t[cls]), torch.max(temp_t[cls])])

                indexing += 1

            p_bar.set_description()
            p_bar.set_postfix(f1_score=np.mean(fscore) / batch, prec=np.mean(prec) / batch,
                              acc=np.mean(acc) / batch, max_label=torch.max(target_label1))

        print('total items: {}'.format(total_items))
        aji_scores = aji_scores/total_items
        jaccard_scores = jaccard_scores / total_items
        fscores = fscores / total_items
        hds = hds/total_items
        accuracies = accuracies / total_items
        senstivities = senstivities / total_items
        precisions = precisions / total_items
        cm = np.round(cm / total_items).astype(int)
        print('Average AJI: {}'.format(aji_scores))
        print('class jaccard scores: {} and Average jaccard score: {}'.format(jaccard_scores,
                                                                              np.mean(jaccard_scores)))
        print('class F1 scores: {} and Average F1 score: {}'.format(fscores, np.mean(fscores)))
        print('class Hausdorff Distances: {} and Average Hausdorff Distance: {}'.format(hds, np.mean(hds)))
        print('class accuracies: {} and Average accuracy: {}'.format(accuracies, np.mean(accuracies)))
        print('class sensitivities: {} and Average sensitivity: {}'.format(senstivities, np.mean(senstivities)))
        print('class precisions: {} and Average precision: {}'.format(precisions, np.mean(precisions)))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='Blues', values_format='')
        plt.savefig('./cm_' + model_type + '_' + loss_type + '_' + str(BATCH_SIZE) + '.jpg', dpi=300)


def main():
    backbone = 'Unet'

    # Loading the data
    img1_list = get_images_list(TEST_IMAGE_DIR)
    label_file = loadmat(TEST_LABEL_PATH)
    train_labels = label_file['data']

    test_dataset = Histo_Dataset(TEST_IMAGE_DIR, img1_list, train_labels,
                                 transform=transformations)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, drop_last=False, num_workers=4)

    path_train = "path for the downstream train checkpoint"
    snapshot = torch.load(path_train)
    print(DEVICE)
    print(path_train)

    model = SegNet(dim=64, channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(snapshot['model_state_dict'])

    eval_epoch(test_loader, model, NUM_CLASSES, DEVICE)


if __name__ == '__main__':
    main()
