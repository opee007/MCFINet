from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import random
import torch
from imgaug import augmenters as iaa
import imgaug
from PIL import Image
imgaug.imgaug.seed(1)

class VaihingenSegmentation(Dataset):

    NUM_CLASSES = 6

    def __init__(self,
                 args,
                 base_dir='/Vaihingen_dirs/',
                 split='val'):
        super().__init__()
        self._base_dir = base_dir
        self.split = split
        self.args = args
        if self.split == 'train'or self.split == 'train_val':
            self._image_dir = os.path.join(self._base_dir, 'train_data', '512_top')
            self._cat_dir = os.path.join(self._base_dir, 'train_data', '512_label')
            self.images = sorted(os.listdir(self._image_dir))#[:30] # 2695
            self.cats = sorted(os.listdir(self._cat_dir))#[:30]
        if self.split == 'val':
            self._image_dir = os.path.join(self._base_dir, 'validation_data', '512_top')
            self._cat_dir = os.path.join(self._base_dir, 'validation_data', '512_label')
            self.images = sorted(os.listdir(self._image_dir)) # 2695
            self.cats = sorted(os.listdir(self._cat_dir))
        self.categories = [
            'Imps',
            'Building',
            'Lowvg',
            'Tree',
            'Car',
            'bg'
            ]

    def __len__(self):
        # print(len(self.images))
        return len(self.images)
        # return 10
    def _make_img_gt_point_pair(self, index):
        _img_path = os.path.join(self._image_dir, self.images[index])
        _target_path = os.path.join(self._cat_dir, self.cats[index])
        _img = cv2.imread(_img_path)
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _target = cv2.imread(_target_path, cv2.IMREAD_GRAYSCALE)

        return _img, _target

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            sample = self.img_aug(sample, self.args)

        sample = self.ToTensor(sample)
        return sample

    def img_aug(self,sample, args):
        img, label = sample['image'], sample['label']
        flipper = iaa.Fliplr(0.5).to_deterministic()
        label = flipper.augment_image(label)
        img = flipper.augment_image(img)

        vflipper = iaa.Flipud(0.5).to_deterministic()
        img = vflipper.augment_image(img)
        label = vflipper.augment_image(label)
        if random.random() < 0.5:
            rot_time = random.choice([1, 2, 3])
            img = np.rot90(img, rot_time)
            label = np.rot90(label, rot_time)
        if random.random() < 0.5:
            translater = iaa.Affine(  # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=random.randint(0, 90),
                scale={"x": (1, 1.4), "y": (1, 1.4)},
                shear=(-8, 8),
                mode='symmetric'#

            ).to_deterministic()
            img = translater.augment_image(img)
            label = translater.augment_image(label)
        img = cv2.resize(img, (args.crop_size, args.crop_size))
        label = cv2.resize(label, (args.crop_size, args.crop_size))
        sample['image'], sample['label'] = img, label

        return sample

    def ToTensor(self,sample):
        img, label = sample['image'], sample['label']
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32) / 255.0)
        label = torch.from_numpy(label.astype(np.float32))
        sample['image'], sample['label'] = img, label

        return sample

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512

    Vaihingen_train = VaihingenSegmentation(args, split='train')

    dataloader = DataLoader(Vaihingen_train, batch_size=5, shuffle=True, num_workers=8)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            print(gt.max())
            gt = np.array(gt[jj]).astype(np.uint8)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(gt*50)

        if ii == 1:
            break

    plt.show(block=True)
