from os.path import join,splitext
from os import listdir
from torch.utils.data import Dataset
from torchvision.transforms import transforms as tfs
from PIL import Image
import torchvision.transforms.functional as F
import logging
import glob
import cv2
import random


class SuperviselyDataset(Dataset):

    def __init__(self, data_root):
        self.name = 'supervisely'
        self.img_root = join(data_root, self.name, 'himgs')
        self.mask_root = join(data_root, self.name, 'hmasks')
        self.data = [file for file in listdir(self.img_root) if not file.startswith('.')]
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
        pass

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
        basefile_name = splitext(self.data[i])
        mask_file_name = basefile_name[0] + '_matte' + basefile_name[1]
        mask_file = join(self.mask_root, mask_file_name)
        img_file = join(self.img_root, file_name)

        img = Image.open(img_file)
        mask = Image.open(mask_file)
        img, mask = self.rand_crop(img, mask, (300, 300))
        return img, mask


class IsbiDataset(Dataset):

    def __init__(self, data_root):
        self.name = 'ISBI'
        self.data_path = join(data_root, self.name, 'train')
        self.imgs_path = glob.glob(join(self.data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)
