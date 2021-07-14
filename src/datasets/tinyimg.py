# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import os


class SUBTINYIMG(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform = None, target_transform = None, train=True):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        """
        Gets the requested element from the datasets.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        not_aug_img = self.not_aug_transform(original_img)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


def split_by_task(targets, task_num, class_per_task, test_prec = 0.1):
    masks = []
    train_masks = []
    test_masks = []
    for i in range(task_num):
        masks.append([])
    for i in range(len(targets)):
        v = targets[i] // class_per_task
        masks[v].append(i)
    for i in range(task_num):
        mask = np.array(masks[i])
        np.random.shuffle(mask)
        l = mask.shape[0]
        thd = int(l * test_prec)
        test_masks.append(mask[:thd])
        train_masks.append(mask[thd:])
    return train_masks, test_masks


class SeqTinyImg:
    NAME = 'seq-tinyimg'
    N_TASKS = 10
    N_CLASSES = 200
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4802, 0.4480, 0.3975),
                              (0.2770, 0.2691, 0.2821))])

    def __init__(self, args):
        self.args = args
        self.root = args['dataset']['root']
        self.sub_train_datasets = []
        self.sub_test_datasets = []
        self.build_sub_datasets()

    def load_tinyimg_data(self, root):
        if os.path.isdir(root) and len(os.listdir(root)) > 0:
            print('Download not needed, files already on disk.')
        else:
            from google_drive_downloader import GoogleDriveDownloader as gdd

            # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
            print('Downloading dataset')
            gdd.download_file_from_google_drive(
                file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',
                dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                unzip=True)

        data = []
        targets = []
        for num in range(20):
            data.append(np.load(os.path.join(root, 'processed/x_train_{:02d}.npy'.format(num+1))))
            #data.append(np.load(os.path.join(root, 'processed/x_val_{:02d}.npy'.format(num+1))))
            targets.append(np.load(os.path.join(root, 'processed/y_train_{:02d}.npy'.format(num + 1))))
            #targets.append(np.load(os.path.join(root, 'processed/y_val_{:02d}.npy'.format(num + 1))))

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        return data, targets

    def build_sub_datasets(self):
        class_per_task = self.N_CLASSES // self.N_TASKS
        train_transform = self.TRANSFORM
        test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        data, targets = self.load_tinyimg_data(os.path.join(self.root, 'TINYIMG'))

        train_masks, test_masks = split_by_task(targets, self.N_TASKS, class_per_task)

        for i in range(self.N_TASKS):
            if self.args['model']['type'] == 'joint':
                if i == self.N_TASKS - 1:
                    total_mask = np.concatenate(train_masks)
                    self.sub_train_datasets.append(
                        SUBTINYIMG(data=data[total_mask], targets=targets[total_mask],
                                   transform=train_transform, train=True))
            else:
                self.sub_train_datasets.append(
                    SUBTINYIMG(data=data[train_masks[i]], targets=targets[train_masks[i]],
                               transform=train_transform, train=True))
            self.sub_test_datasets.append(
                SUBTINYIMG(data=data[test_masks[i]], targets=targets[test_masks[i]],
                           transform=test_transform, train=False))

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SeqTinyImg.TRANSFORM])
        return transform
