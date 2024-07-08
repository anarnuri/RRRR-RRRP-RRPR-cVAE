import torch
from torch.utils.data import Dataset
import glob
import os 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from constants import * 
import random

class SingleDataset(Dataset):
    def __init__(self, transform):
        self.imgs_path = 'coupler_curves\\'
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        self.transform = transform
        self.classes = {'RRRR':0, 'RRRP':1, 'RRPR':2}
        
        for class_path in file_list[:100]:
            class_name = class_path.split("\\")[-1]
            img_paths = os.listdir(class_path)
            
            for img_path in img_paths:
                img_path = class_path + '\\' + img_path
                self.data.append([img_path, class_name])

        random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]

        img = cv2.imread(img_path, 0)
        img = cv2.bitwise_not(img)

        img_tensor = self.transform(img)
        name = img_path.split('\\')[-1].split(' ')
        
        label =  torch.FloatTensor([self.classes[class_name]])

        joints = torch.FloatTensor([float(name[0]), float(name[1]),
                                    float(name[2]), float(name[3]),
                                    float(name[4]), float(name[5]),
                                    float(name[6]), float(name[7][:-4])]) / 3       

        return img_tensor, joints, label