import cv2
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from PIL import Image
from dataset.transform import*
from torchvision import transforms
from copy import deepcopy

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

    
###############################  semi train dataset  ###########################
class DriveDataset(Dataset):
    def __init__(self, base_dir, list_name,image_size,dataset,transform):
        self.base_dir = base_dir
        self.sample_list = []
        self.h, self.w = image_size
        self.dataset = dataset
        self.transform = transform
        with open(self.base_dir + list_name, 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '')  for item in self.sample_list]
        #self.sample_list = self.sample_list[0:700]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        if self.dataset == 'skin':
            image = cv2.imread(self.base_dir + '/images/'+name+'.jpg', cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(self.base_dir + '/images/'+name+'.png', cv2.IMREAD_COLOR)
            
        label = cv2.imread(self.base_dir + '/masks/'+name+'.png', cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        weak_image, weak_mask = weak_augment(image, label, prob=0.5)
        # import ipdb;ipdb.set_trace()
        # 强增强
        strong_image, strong_mask = strong_augment(image, label, prob=0.5)
        weak_image = np.asarray(weak_image, np.float32).transpose((2, 0, 1))
        strong_image = np.asarray(strong_image, np.float32).transpose((2, 0, 1))
        sample = {
            "weak_image": weak_image,
            "weak_mask": weak_mask,
            "strong_image": strong_image,
            "strong_mask": strong_mask,
        }
        
        
        sample = self.transform(sample)

        return sample

###############################  semi train patch  ###########################
class DriveDataset_patch(Dataset):
    def __init__(self, base_dir, list_name, image_size, patch_size=224, overlap=112, dataset=None, transform=None):
        self.base_dir = base_dir
        self.sample_list = []
        self.h, self.w = image_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.dataset = dataset
        self.transform = transform
        with open(self.base_dir + list_name, 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list) * ((self.h - self.overlap) // (self.patch_size - self.overlap)) * ((self.w - self.overlap) // (self.patch_size - self.overlap))
    #320个patch
    def __getitem__(self, idx): #idx是总数据集中的第 idx 个 patch，最大为319
        patches_per_row = (self.h - self.overlap) // (self.patch_size - self.overlap)
        patches_per_col = (self.w - self.overlap) // (self.patch_size - self.overlap)
        patches_per_image = patches_per_row * patches_per_col
        # 获取图片名称
        img_idx = idx // patches_per_image #图片索引，第几张图片
        name = self.sample_list[img_idx] #'1'

        # 加载并调整图片大小
        image = cv2.imread(self.base_dir + 'images/' + name + '.png', cv2.IMREAD_COLOR)
        label = cv2.imread(self.base_dir + 'masks/' + name + '.png', cv2.IMREAD_GRAYSCALE)

        # 将图片调整为目标尺寸
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST) / 255

         # 计算 patch 在行和列上的起始位置
        patch_row = ((idx % patches_per_image) // patches_per_row) * (self.patch_size - self.overlap)
        patch_col = (idx % patches_per_image % patches_per_row) * (self.patch_size - self.overlap)
        
        # 切割图像和标签的patch
        image_patch = image[patch_row:patch_row + self.patch_size, patch_col:patch_col + self.patch_size]
        label_patch = label[patch_row:patch_row + self.patch_size, patch_col:patch_col + self.patch_size]
        # 调试打印每个patch的形状
        # print(f"Image patch shape: {image_patch.shape}")
        # print(f"Label patch shape: {label_patch.shape}")
        # 数据增强（可选）
        if random.random() > 0.5:
            image_patch, label_patch = random_rot_flip(image_patch, label_patch)
        #(224, 224, 3) <class 'numpy.ndarray'>
        # 转换为tensor格式
        image_patch = np.asarray(image_patch, np.float32).transpose((2, 0, 1))#(3, 224, 224) <class 'numpy.ndarray'>
        # import ipdb;ipdb.set_trace() 
        sample = {'image': image_patch, 'label': label_patch}
        sample = self.transform(sample)
        sample["idx"] = name
        
        return sample


###############################  semi train patch augment ###########################
def weak_augment(image, mask, prob=0.5):
    """
    弱增强：包括水平翻转、裁剪
    """
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    # 水平翻转
    image, mask = hflip(image, mask, p=prob)#PIL.Image.Image
    # image, mask = resize(image, mask, (0.5, 2.0))
    image = np.array(image)
    mask = np.array(mask)

    return image, mask

def strong_augment(image, mask, prob=0.8):
    """
    强增强：颜色抖动，随机灰度，随机模糊
    """
    # 强增强流程
    # 如果输入是 numpy.ndarray，将其转换为 PIL.Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    if random.random() < prob:
        image = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image)
    image = transforms.RandomGrayscale(p=0.2)(image)
    image = blur(image, p=0.5)
    image = np.array(image)
    return image, mask
def adjust_contrast(image, alpha_range=(0.8, 1.2)):
    """
    随机调整图像对比度，避免过度增强
    :param image: 输入图像
    :param alpha_range: 对比度因子范围 (tuple)
    :return: 调整后的图像
    """
    alpha = np.random.uniform(*alpha_range)  # 从范围中随机选择一个值
    mean = np.mean(image)
    adjusted = mean + alpha * (image - mean)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


class DriveDataset_patch_augment(Dataset):
    def __init__(self, base_dir, list_name, image_size, patch_size=224, overlap=112, dataset=None, transform=None, weak_prob=0.5, strong_prob=0.5):
        self.base_dir = base_dir
        self.sample_list = []
        self.h, self.w = image_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.dataset = dataset
        self.transform = transform
        self.weak_prob = weak_prob  # 弱增强概率
        self.strong_prob = strong_prob  # 强增强概率
        with open(self.base_dir + list_name, 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list) * ((self.h - self.overlap) // (self.patch_size - self.overlap)) * ((self.w - self.overlap) // (self.patch_size - self.overlap))

    def __getitem__(self, idx):
        patches_per_row = (self.h - self.overlap) // (self.patch_size - self.overlap)
        patches_per_col = (self.w - self.overlap) // (self.patch_size - self.overlap)
        patches_per_image = patches_per_row * patches_per_col
        
        img_idx = idx // patches_per_image
        name = self.sample_list[img_idx]
        if self.dataset == 'drive':
            image = cv2.imread(self.base_dir + 'images/' + name + '.png', cv2.IMREAD_COLOR)
            label = cv2.imread(self.base_dir + 'masks/' + name + '.png', cv2.IMREAD_GRAYSCALE)
        elif self.dataset == 'hrf':
            image = cv2.imread(self.base_dir + 'images/' + name + '.jpg', cv2.IMREAD_COLOR)
            label = cv2.imread(self.base_dir + 'masks/' + name + '.jpg', cv2.IMREAD_GRAYSCALE)
        elif self.dataset == 'Kvasir-SEG':
            image = cv2.imread(self.base_dir + 'images/' + name + '.jpg', cv2.IMREAD_COLOR)
            label = cv2.imread(self.base_dir + 'masks/' + name + '.jpg', cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(self.base_dir + 'images/' + name + '.png', cv2.IMREAD_COLOR)
            label = cv2.imread(self.base_dir + 'masks/' + name + '.png', cv2.IMREAD_GRAYSCALE)   

        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST) / 255

        patch_row = ((idx % patches_per_image) // patches_per_row) * (self.patch_size - self.overlap)
        patch_col = (idx % patches_per_image % patches_per_row) * (self.patch_size - self.overlap)
        
        image_patch = image[patch_row:patch_row + self.patch_size, patch_col:patch_col + self.patch_size]
        label_patch = label[patch_row:patch_row + self.patch_size, patch_col:patch_col + self.patch_size]

        # 弱增强
        # weak_image, weak_mask = image_patch, label_patch
        weak_image, weak_mask = weak_augment(image_patch, label_patch, prob=self.weak_prob)
        # import ipdb;ipdb.set_trace()
        # 强增强
        strong_image, strong_mask = strong_augment(image_patch, label_patch, prob=self.strong_prob)
        strong_image, strong_mask = image_patch, label_patch
        weak_image = np.asarray(weak_image, np.float32).transpose((2, 0, 1))
        strong_image = np.asarray(strong_image, np.float32).transpose((2, 0, 1))
        sample = {
            "weak_image": weak_image,
            "weak_mask": weak_mask,
            "strong_image": strong_image,
            "strong_mask": strong_mask,
        }
        
        
        sample = self.transform(sample)

        return sample


class TwoStreamBatchSampler(Sampler):

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices #80 5*16
        self.secondary_indices = secondary_indices #240  15*16
        self.secondary_batch_size = secondary_batch_size #8
        self.primary_batch_size = batch_size - secondary_batch_size #8
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self): 
        primary_iter = iterate_once(self.primary_indices) 
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            #secondary_batch + primary_batch
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable) 


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

################################  test dataset  ###########################
class testBaseDataSets(Dataset):
    def __init__(self, base_dir, list_name,image_size,dataset,transform):
        self.base_dir = base_dir
        self.sample_list = []
        self.h, self.w = image_size
        self.dataset = dataset
        self.transform = transform
        with open(self.base_dir + list_name, 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '')  for item in self.sample_list]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        if self.dataset == 'drive':
            image = cv2.imread(self.base_dir + 'images/'+name+'.png', cv2.IMREAD_COLOR)         
            label = cv2.imread(self.base_dir + 'masks/'+name+'.png', cv2.IMREAD_GRAYSCALE)  
        elif self.dataset in ['la']:
            image = cv2.imread(self.base_dir + '/images/'+name+'.png', cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image, axis=-1)  # [H, W] -> [H, W, 1]        
            label = cv2.imread(self.base_dir + 'masks/' + name + '.png', cv2.IMREAD_GRAYSCALE)
            label = label/255.0
        elif self.dataset == 'hrf':
            image = cv2.imread(self.base_dir + 'images/' + name + '.jpg', cv2.IMREAD_COLOR)
            label = cv2.imread(self.base_dir + 'masks/' + name + '.jpg', cv2.IMREAD_GRAYSCALE)
        elif self.dataset == 'Kvasir-SEG':
            image = cv2.imread(self.base_dir + 'images/' + name + '.jpg', cv2.IMREAD_COLOR)
            label = cv2.imread(self.base_dir + 'masks/' + name + '.jpg', cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(self.base_dir + 'images/' + name + '.jpg', cv2.IMREAD_COLOR)
            label = cv2.imread(self.base_dir + 'masks/' + name + '.png', cv2.IMREAD_GRAYSCALE)
        # image = adjust_contrast(image, alpha_range=(0.7, 0.7))
        
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        return sample


class RandomGenerator(object):
    def __init__(self):
        self.k = 9
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).long()
        sample = {'image': image, 'label': label}
        return sample


class RandomGenerator1(object):
    def __init__(self):
        self.k = 9
    def __call__(self, sample):
        weak_img, weak_mask, strong_img, strong_mask = sample['weak_image'], sample['weak_mask'], sample['strong_image'], sample['strong_mask']
        weak_img = torch.from_numpy(weak_img.astype(np.float32))
        strong_img = torch.from_numpy(strong_img.astype(np.float32))
        weak_mask = torch.from_numpy(weak_mask.astype(np.uint8)).long()
        strong_mask = torch.from_numpy(strong_mask.astype(np.uint8)).long()
        sample = {'weak_image': weak_img, 'weak_mask': weak_mask, 'strong_image':strong_img, 'strong_mask':strong_mask}
        return sample
