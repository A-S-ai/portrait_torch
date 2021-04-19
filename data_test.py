from os import listdir
from os.path import join
import numpy as np
import pylab as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms as tfs
import logging
from PIL import Image


dir_img = 'data/img1/'
dir_mask = 'data/mask1/'


class BasicDataset(Dataset):

    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [file for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        pil_img = pil_img.numpy()
        pil_img = pil_img.transpose((1, 2, 0))
        print(pil_img.shape)
        plt.imshow(pil_img)

        if len(pil_img.shape) == 2:
            pil_img = np.expand_dims(pil_img, axis=2)

        # HWC to CHW
        img_trans = pil_img.transpose((2, 0, 1))
        #图像归一化
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        plt.show()
        return img_trans

    def rand_crop(self, img, mask):
        img = tfs.PILToTensor()(img)
        mask = tfs.PILToTensor()(mask)
        i, j, h, w = tfs.RandomResizedCrop.get_params(img, [0.5, 1.0], [0.75, 1.33])
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        img = TF.resize(img, [300, 300])
        mask = TF.resize(mask, [300, 300])
        return img, mask

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = join(self.masks_dir,idx)
        img_file = join(self.imgs_dir,idx)

        mask = Image.open(mask_file)
        img = Image.open(img_file)

        img, mask = self.rand_crop(img, mask)
        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

dataset = BasicDataset(dir_img, dir_mask, 1)
print(dataset[0])