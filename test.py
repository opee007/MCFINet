import torch
import numpy as np
import cv2
from tqdm import tqdm
import os
from modeling.deeplab import DeepLab
from modeling.MCFINet import MCFINet
from modeling.pspnet import pspnet
from modeling.fcn import fcn8s
from modeling.unet import unet
from modeling.segnet import segnet
from modeling.enet import ENet
from modeling.bisenet import BiSeNet
import time
import natsort
import glob
from libtiff import TIFF
from skimage.io import imread, imsave
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
output_name = [
    'area2',
    'area4',
    'area6',
    'area8',
    'area10',
    'area12',
    'area14',
    'area16',
    'area20',
    'area22',
    'area24',
    'area27',
    'area29',
    'area31',
    'area33',
    'area35',
    'area38'
]
palette = [(255, 255, 255), # 白色-背景
           (0, 0, 255), # 绿色-树木
           (0, 255, 0), # 蓝色-建筑物
           (255, 0, 0), # 红色-杂物
           (0, 255, 255), # 青色-草地
           (255, 255, 0)] # 黄色-汽车

TEST_SET = natsort.natsorted(glob.glob('/mnt/wdj/dataset/isprs/Vaihingen/test/*tif'))#[:10]
print(TEST_SET)


image_size = 512
stride = 448

def get_oneimg(crop, model):
    # crop = torch.from_numpy(crop)
    crop = crop.transpose(2, 0, 1)
    crop = torch.from_numpy(crop.astype(np.float32) / 255.0)
    with torch.no_grad():
        crop = crop.unsqueeze(0).cuda()
        crop = crop.cuda()
        pred = model(crop)
        pred = pred[0]
        pred = torch.argmax(pred.data.cpu(), 0).byte().numpy()  # 512*512

    pred = cv2.resize(pred, (image_size, image_size)).astype(np.uint8)
    color_image = np.array(palette)[pred.ravel()].reshape(
        pred.shape + (3,))
    return color_image

def get_imglist(img):
    img_list = []
    img_list.append(img)
    img_list.append(np.flipud(img).copy())
    img_list.append(np.fliplr(img).copy())
    img_list.append(np.rot90(img, 1).copy())
    img_list.append(np.rot90(img, 2).copy())
    img_list.append(np.rot90(img, 3).copy())
    return img_list

def get_onepred(crop, model):
    crop = crop.transpose(2, 0, 1)
    crop = torch.from_numpy(crop.astype(np.float32) / 255.0)
    with torch.no_grad():
        crop = crop.unsqueeze(0).cuda()
        crop = crop.cuda()
        pred = model(crop)
        pred = pred[0]
    return pred.data.cpu().permute(1, 2, 0).numpy()

def restore_mask(img_list):
    img_list[0] = img_list[0]
    img_list[1] = np.flipud(img_list[1])
    img_list[2] = np.fliplr(img_list[2])
    img_list[3] = np.rot90(img_list[3], 3)
    img_list[4] = np.rot90(img_list[4], 2)
    img_list[5] = np.rot90(img_list[5], 1)
    return img_list

def decode_mask(mask_list):
    mask = mask_list[0]
    for i in range(1, len(mask_list)):
        mask += mask_list[i]
    mask = mask/len(mask_list)
    return mask
rate_list = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# rate_list = [0.75, 1.0, 1.25]

def test_aug(crop, model):
    res_list = []
    for rate in rate_list:
        tem_crop = cv2.resize(crop, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        crop_list = get_imglist(tem_crop)
        res_list2 = []
        for one_img in crop_list:
            pred = get_onepred(one_img, model)
            res_list2.append(pred)
        res_list2 = restore_mask(res_list2)
        res_t = decode_mask(res_list2)
        res_t = cv2.resize(res_t, (512, 512))
        res_list.append(res_t)
    res = decode_mask(res_list)
    res = np.argmax(res, axis=-1)
    color_image = np.array(palette)[res.ravel()].reshape(
        res.shape + (3,))
    return color_image

model = MCFINet(num_classes=6,
                backbone='resnet-101',
                output_stride=32,
                )
# model = DeepLab()
model = torch.nn.DataParallel(model, device_ids=[0])

checkpoint = torch.load('./run/isprs_Vaihingen/MCFINet+SEBlock/experiment_0/checkpoint80.pth.tar') #flip/model_best.pth.tar')
model.module.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()

use_augtest = True

def predict():
    st = time.time()
    print(st)
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        print(path)
        img = cv2.imread(path)
        # print(img)
        h, w, _ = img.shape
        h, w, _ = img.shape
        print(h, w)
        img = img[:, :, ::-1]
        # padding_h = h + image_size
        # padding_w = w + image_size
        # padding_img = np.zeros((h, w, 3), dtype=np.uint8)
        # padding_img[0:h, 0:w, :] = img[:, :, :]
        # padding_img = padding_img[:, :, ::-1]
        mask = np.zeros((h, w, 3), np.uint8)

        for i in tqdm(range(0, h, stride)):
            for j in range(0, w, stride):
                if i == 0:
                    if j == 0:
                        crop = img[i:i + image_size, j:j + image_size, :]
                        if use_augtest:
                            color_image = test_aug(crop, model)
                        else:
                            color_image = get_oneimg(crop, model)
                        mask[i:i + image_size, j:j + image_size, :] = color_image[:, :, :]
                    elif j+image_size < w:
                        crop = img[i:i + image_size, j:j + image_size, :]
                        if use_augtest:
                            color_image = test_aug(crop, model)
                        else:
                            color_image = get_oneimg(crop, model)
                        mask[i:i + image_size, j+32:j + image_size, :] = color_image[:, 32:, :]
                    else:
                        crop = img[i:i + image_size, w-image_size:w, :]
                        if use_augtest:
                            color_image = test_aug(crop, model)
                        else:
                            color_image = get_oneimg(crop, model)
                        mask[i:i + image_size, w-image_size+32:w, :] = color_image[:, 32:, :]
                elif i+image_size < h:
                    if j == 0:
                        crop = img[i:i + image_size, j:j + image_size, :]
                        if use_augtest:
                            color_image = test_aug(crop, model)
                        else:
                            color_image = get_oneimg(crop, model)
                        mask[i+32:i + image_size, j:j + image_size, :] = color_image[32:, :, :]
                    elif j+image_size < w:
                        crop = img[i:i + image_size, j:j + image_size, :]
                        if use_augtest:
                            color_image = test_aug(crop, model)
                        else:
                            color_image = get_oneimg(crop, model)
                        mask[i+32:i + image_size, j+32:j + image_size, :] = color_image[32:, 32:, :]
                    else:
                        crop = img[i:i + image_size, w-image_size:w, :]
                        if use_augtest:
                            color_image = test_aug(crop, model)
                        else:
                            color_image = get_oneimg(crop, model)
                        mask[i+32:i + image_size, w-image_size+32:w, :] = color_image[32:, 32:, :]
                else: #  when i+image_size > h
                    if j == 0:
                        crop = img[h-image_size:h, j:j + image_size, :]
                        if use_augtest:
                            color_image = test_aug(crop, model)
                        else:
                            color_image = get_oneimg(crop, model)
                        mask[h-image_size+32:h, j:j + image_size, :] = color_image[32:, :, :]
                    elif j+image_size < w:
                        crop = img[h-image_size:h, j:j + image_size, :]
                        if use_augtest:
                            color_image = test_aug(crop, model)
                        else:
                            color_image = get_oneimg(crop, model)
                        mask[h-image_size+32:i + image_size, j+32:j + image_size, :] = color_image[32:, 32:, :]
                    else:
                        crop = img[h-image_size:h, w-image_size:w, :]
                        if use_augtest:
                            color_image = test_aug(crop, model)
                        else:
                            color_image = get_oneimg(crop, model)
                        mask[h-image_size+32:h, w-image_size+32:w, :] = color_image[32:, 32:, :]

        imsave('./reslut/mcfinet+seblock/' + 'top_mosaic_09cm_'+ output_name[n] + '_class.tif', mask[0:h, 0:w, :][:, :, ::-1])
    print(time.time() - st)

if __name__ == '__main__':
    with torch.no_grad():
    	predict()

'''
MDCM:4.02
deeplab v+: 4.04       63
pspnet: 4.53           76
fcn8s: 5.41  all time: 85.17
unet: 3.90   all time: 58 
segnet: 4.11 all time: 58.3
enet:        all tiem: 55
'''