import pickle
import os
import cv2
import matplotlib.pyplot as plt 

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils_cus import ResizeLongestSide


class RadSamDataset(Dataset):

    def __init__(self, 
                 img_dir, 
                 masks_path, 
                 ptns_path, 
                 start_idx, 
                 end_idx,
                 encoder_img_size=1024,
                 device='cpu'
        ):
        self.img_dir = img_dir
        self.masks_path = masks_path
        self.pnts_path = ptns_path
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.encoder_img_size = encoder_img_size
        self.transform = ResizeLongestSide(encoder_img_size)
        self.device = device

        # HARD CODED
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        self.pixel_mean = torch.Tensor(self.pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(self.pixel_std).view(-1, 1, 1)
        
        # load pickle masks
        with open(self.masks_path, "rb") as f:
            self.masks = pickle.load(f)

        with open(self.pnts_path, "rb") as f:
            self.pnts = pickle.load(f)

        # map idx to img idx
        img_bins = [0]
        for mask in self.masks:
            img_bins.append(len(mask) + img_bins[-1])
        self.img_bins = img_bins

        # convert to arrays
        self.masks = np.concatenate(self.masks, axis=0)
        self.pnts = np.concatenate(self.pnts, axis=0)

    def img_idx(self, idx):
        return np.digitize(idx, self.img_bins) - 1
    
    def verif(self, idx, save=False):
        assert 0 <= idx <= len(self.masks) - 1, f"idx 0 and {len(self.masks) - 1}"
 
        # get image
        img_idx = self.img_idx(idx)
        print(img_idx)
        img_path = os.path.join(self.img_dir, f"{self.start_idx + img_idx}.png")
        img = cv2.imread(img_path)

        # get binary masks and pnt 
        img_mask = self.masks[idx]
        pnt = self.pnts[idx]

        # plot
        plt.imshow(img)
        seg_masked = np.ma.masked_where(img_mask == 0, img_mask)
        plt.imshow(seg_masked, cmap="Paired")
        y, x = pnt
        plt.scatter(x, y, marker='x', c='red')
        plt.axis("on")
        if save:
            plt.savefig(f'img{self.start_idx + img_idx}_maskidx{idx}.png')
    
    def __len__(self):
        return len(self.masks)
    
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.encoder_img_size - h
        padw = self.encoder_img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def __getitem__(self, idx):

        img_idx = self.img_idx(idx)

        # input image
        img_n = self.start_idx + img_idx
        img_n_t = torch.as_tensor([img_n], dtype=torch.int, device=self.device)

        img_path = os.path.join(self.img_dir, f"{img_n}.png")
        input_image = cv2.imread(img_path)
        input_image = self.transform.apply_image(input_image)
        input_image_torch = torch.as_tensor(input_image)
        # input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        input_image = self.preprocess(input_image_torch)

        # point
        orig_coords = self.pnts[idx]
        orig_coords_t = torch.as_tensor(orig_coords[None, :].astype(np.int16), dtype=torch.float, device=self.device)
        point_coords = orig_coords[::-1]  # needs to be x, y and not y, x
        point_coords = point_coords[None, :]
        point_coords = self.transform.apply_coords(point_coords, (512, 512))
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
        labels_torch = torch.as_tensor([1], dtype=torch.int, device=self.device)

        # mask
        mask = self.masks[idx]

        return input_image, (coords_torch, labels_torch), mask, (img_n_t, orig_coords_t)


        



        


        
        
