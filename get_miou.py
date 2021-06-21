# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:47:41 2018

@author: wdj
"""

import numpy as np
import glob
import cv2
import natsort
from tqdm import tqdm
import numpy as np
import os

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(np.delete(MIoU, 1, axis = 0))
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


images = natsort.natsorted(glob.glob('./reslut/mcfinet+seblock/*.tif'))


def rgb2mask(img):
    h, w, _ = img.shape
    mask = img//255
    maskbs = mask[:, :, 0]
    maskgs = mask[:, :, 1]
    maskrs = mask[:, :, 2]

    maskys = np.multiply(maskrs, maskgs)
    maskcs = np.multiply(maskbs, maskgs)
    maskw = np.multiply(maskys, maskcs)

    maskb = maskbs
    maskb[maskcs == 1] = 0
    maskg = maskgs
    maskg[maskys == 1] = 0
    maskg[maskcs == 1] = 0
    maskr = maskrs
    maskr[maskys == 1] = 0
    masky = maskys
    masky[maskw == 1] = 0
    maskc = maskcs
    maskc[maskw == 1] = 0
#    maskc = maskcs; maskb[maskw == 1] = 0
    temp = np.zeros((h, w)).astype(np.uint8)
    temp = temp + maskr + maskg*2 + maskb*3 + masky*4 + maskc*5
    return temp

evaluator = Evaluator(6)
evaluator.reset()
for i in tqdm(range(len(images))):

    img = cv2.imread(images[i])              # 3 channels
    labpath = os.path.join(
        "assess_classification_reference_implementation/assess_classification_2D_ISPRS_web/ISPRS_semantic_labeling_data_upload/gts/",
        os.path.basename(images[i]).replace("_class", ""))
    imgmask = rgb2mask(img)
    lab = cv2.imread(labpath)  
    labmask = rgb2mask(lab)
    evaluator.add_batch(imgmask, labmask)

mIOU = evaluator.Mean_Intersection_over_Union()
print(mIOU)


