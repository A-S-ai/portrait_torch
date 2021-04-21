from os.path import join
from os import listdir
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import transforms as tfs
import logging
from PIL import Image


class BasicDataset(Dataset):

    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.data = [file for file in listdir(img_dir) if not file.startswith('.')]
        self.img_trans, self.mask_trans = self.default_transform()

        logging.info(f'Creating dataset with {len(self.data)} examples')

    def __len__(self):
        return len(self.data)

    @classmethod
    def default_transform(cls):
        return tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]), tfs.Compose([
            tfs.ToTensor()
        ])

    @classmethod
    def preprocess(cls, img):
        # print(img.shape)
        # img = img.permute(1, 2, 0)
        # print(img.shape)

        # CHW
        # if img.shape[0] == 2:
        #     img = img.unsqueeze(2)

        # HWC to CHW
        # img = img.permute(2, 0, 1)
        # print(img.shape)
        # 图像归一化
        # import matplotlib.pyplot as plt
        # print(type(img))
        # plt.imshow(img.permute(1, 2, 0), cmap='gray')
        # plt.show()
        if img.max() > 1:
            img = img / 255.0
        return img

    def rand_crop(self, img, mask, new_size):
        img = self.img_trans(img)
        mask = self.mask_trans(mask)
        i, j, h, w = tfs.RandomResizedCrop.get_params(img, [0.5, 1.0], [0.75, 1.33])
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        img = F.resize(img, new_size)
        mask = F.resize(mask, new_size)
        return img, mask

    def __getitem__(self, i):
        file_name = self.data[i]
        mask_file = join(self.mask_dir, file_name)
        img_file = join(self.img_dir, file_name)

        img = Image.open(img_file)
        mask = Image.open(mask_file)

        img, mask = self.rand_crop(img, mask, (300, 300))
        # img = self.preprocess(img)
        # mask = self.preprocess(mask)

        return img, mask
