from cmath import nan
import csv
import json
import logging
import os
import sys
import pydicom

from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from skimage import exposure
import torch
from torchvision.transforms import InterpolationMode
from dataset.randaugment import RandomAugment


class MIMIC_Dataset(Dataset):
    def __init__(self, csv_path, labels_path, is_train=True):
        self.ann = json.load(open(csv_path,'r'))
        self.img_path_list = list(self.ann)

        self.labels = pd.read_csv(labels_path, index_col=0)
        
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if is_train:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        else:
            self.transform = transforms.Compose([                        
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])     
        
        self.seg_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224],interpolation=InterpolationMode.NEAREST),
        ])

    def __getitem__(self, index):
        while index < len(self.img_path_list):
            img_path = self.img_path_list[index]
            
            if not os.path.exists(img_path):
                print(f"Warning: Image path {img_path} does not exist. Skipping index {index}.")
                index += 1
                continue

            img = PIL.Image.open(img_path).convert('RGB')  
            image = self.transform(img) 

            class_label = self.labels.iloc[index].drop('study_id').values

            return {
                "image": image,
                "label": class_label
                }


    def __len__(self):
        return len(self.img_path_list)
