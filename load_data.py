import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_data()
    return data

def read_table(path):
    return(pd.read_excel(path).values) # default to first sheet

def white0(image, threshold=0):
    "先使用整个数据集的均值和方差来进行实验，理论上每个样本的均值是不同的，同样方差也不同，能够体现样本的一些属性"
    factor = (0.5, 0.5, 0.5)

    new_image = zoom(image, factor, order=0)
    new_image = new_image.astype(np.float32)
    mask = (new_image > threshold).astype(int)
    image_h = new_image * mask
    image_l = new_image * (1 - mask)
    all_mean = 240.7564
    all_std = 76.1665
    ret = (image_h - all_mean) / all_std + image_l

    return ret

def transforms_img(img):
    img_out = np.zeros(img.shape) + img
    if np.random.randint(-5, 5) > 0:
        off_z = np.random.randint(-5, 5)
        off_x = np.random.randint(-5, 5)
        off_y = np.random.randint(-5, 5)
        # print("img_out.shape:", img_out.shape)
        z, x, y = img.shape[0], img.shape[1], img.shape[2]
        if off_z> 0:
            img_array = np.zeros(img_out.shape)
            img_array[:z - off_z, :, :] = img_out[off_z:z, :, :]
            img_out = img_array
        if off_z < 0:
            img_array = np.zeros(img_out.shape)
            img_array[(0-off_z):z, :, :] = img_out[:z - (0-off_z), :, :]
            img_out = img_array
        if off_x > 0:
            img_array = np.zeros(img_out.shape)

            img_array[:, :x - off_x, :] = img_out[:, off_x:x, :]
            img_out = img_array
        if off_x < 0:
            img_array = np.zeros(img_out.shape)
            img_array[:, (0-off_x):x, :] = img_out[:, :x - (0-off_x), :]
            img_out = img_array
        if off_y > 0:
            img_array = np.zeros(img_out.shape)
            img_array[:, :, :y - off_y] = img_out[:, :, off_y:y]

            img_out = img_array
        if off_y < 0:
            img_array = np.zeros(img_out.shape)
            img_array[:, :, (0-off_y):y] = img_out[:, :, :y - (0-off_y)]
            img_out = img_array

    return img_out

class IMG_Folder(torch.utils.data.Dataset):
    def __init__(self,excel_path, data_path, loader=nii_loader,transforms=None):
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform = transforms

    def __len__(self):
        print("len(self.sub_fns):", len(self.sub_fns))
        return len(self.sub_fns)

    def __getitem__(self,index):
        sub_fn = self.sub_fns[index]
        for f in self.table_refer:
            
            sid = str(f[0])
            slabel = (f[1])
            smale = f[2]
            if sid not in sub_fn:
                continue
            sub_path = os.path.join(self.root, sub_fn)
            """下面这两行读取了图像数据："""
            img = self.loader(sub_path)
            img = white0(img)
            if self.transform is not None:
                img = self.transform(img)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            img = torch.from_numpy(img).type(torch.FloatTensor)

            break
        return (img, sid, slabel, smale)
