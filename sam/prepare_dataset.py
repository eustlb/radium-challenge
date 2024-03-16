import os
import glob
import cv2
import random
import numpy as np
import pickle 
import glob
import pandas as pd
from tqdm import tqdm


def load_dataset(dataset_dir):
    dataset_list = []
    # Note: It's very important to load the images in the correct numerical order!
    for image_file in sorted(glob.glob(os.path.join(dataset_dir, "*.png")), key=lambda x: int(x.split("/")[-1].replace(".png", ""))):
        dataset_list.append(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return np.stack(dataset_list, axis=0)


def split_masks(mask):
    """
    array (512x512, ) -> (n_classes, 512, 512).

    :param mask: ndarray, shape (512x512,). 
    """
    # classes values, remove background (0)
    cls_values = set(mask)
    cls_values.remove(0)

    cls_mask_l = np.zeros((len(cls_values), 512, 512), dtype=np.uint8)
    for i, cls_v in enumerate(cls_values):
        idxs = np.where(mask == cls_v)[0]
        cls_mask = np.zeros_like(mask, dtype=np.uint8)
        cls_mask[idxs] = 1
        cls_mask_l[i] = cls_mask.reshape((512, 512))

    return cls_mask_l


def erode_boundary(mask):
    """
    array (512, 512) -> array (512, 512) with eroded mask.

    :param mask: ndarray of shape (512, 512), map of 0 and 1
    :return eroded_mask: new mask, eroded.
    """
    n_it = 6  # number of iterations
    kernel = np.ones((3,3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=n_it)

    while len(np.unique(eroded_mask)) < 2:
        n_it -= 1
        if n_it == 0:
            return mask
        eroded_mask = cv2.erode(mask, kernel, iterations=n_it)

    return eroded_mask


def main(x_train_path, 
         y_train_path, 
         saving_dir,
         idx_start=200, 
         idx_end=400
         ):
    """
    str path, str_path
      -> list of splitted masks (n_classes, 512, 512)
         list of random points in an eroded region of the mask
    both list are saved as pickle.
    """
    
    # load images
    print("loading data...")
    data_train = load_dataset(x_train_path)
    data_train = data_train[idx_start: idx_end] 

    # load masks
    labels_train = pd.read_csv(y_train_path, index_col=0).T
    labels_train = labels_train.values[idx_start: idx_end]
    print("data loaded !")

    # keep track orginal img
    idx_to_img = {}
    
    # split into 0-1 masks, erode boundary and select random point
    splitted_masks, pnts = [], []
    print("splitting masks...")
    for mask in tqdm(labels_train):
        s_masks = split_masks(mask)
        img_pnts = np.zeros((len(s_masks), 2), dtype=np.uint16)
        for i, m in enumerate(s_masks):
            eroded_mask = erode_boundary(m)
            # pick a random point in the eroded mask
            coordinates = np.argwhere(eroded_mask == 1)
            idx = random.randint(0, len(coordinates) - 1)
            img_pnts[i] = coordinates[idx]
        splitted_masks.append(s_masks)
        pnts.append(img_pnts)

    m_path = os.path.join(saving_dir, f"masks_{idx_start}_{idx_end}.pkl")
    with open(m_path, "wb") as f:
        pickle.dump(splitted_masks, f)

    p_path = os.path.join(saving_dir, f"pnts_{idx_start}_{idx_end}.pkl")
    with open(p_path, "wb") as f:
        pickle.dump(pnts, f)


if __name__ == "__main__":

    x_train_path = "/Users/eustachelebihan/Development/radium-challenge/data/x-train"
    y_train_path = "/Users/eustachelebihan/Development/radium-challenge/data/y_train.csv"
    saving_dir = "/Users/eustachelebihan/Development/radium-challenge/sam/data"

    main(x_train_path, 
         y_train_path, 
         saving_dir,
         idx_start=40, 
         idx_end=400
    )






