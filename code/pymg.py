import numpy as np
import matplotlib.pyplot as plt
import cv2 as op
from typing import List, Tuple, Union

        
def resize_image(
            img: List[float], 
            size: Union[str, Tuple[int, int]]):
    
    width, height = img.shape[0], img.shape[1]

    if type(size) == tuple:
        width , height = size[0], size[1]

    else:
        if size == 'original':
            size = (height, width)
        elif size == 'half':
            size = (int(height/2), int(width/2))
        else:
            raise ValueError("Unknown value '{}' for size parameter".format(size))

    img = op.resize(img, size)
    
    return img

def normalize_image(
            img: List[float],
            between: Tuple[float, float]):
    
    x, y = between[0], between[1]
    min_p, max_p = np.min(img), np.max(img)
    img = (img - min_p)/(max_p - min_p)

    img = ((y - x) * img) + x
    return img

def change_view(image_array, view):
    if view == 'CHW':
        return np.transpose(image_array, (2, 0, 1))
    elif view == 'HWC':
        return np.transpose(image_array, (0, 1, 2))
    else:
        raise ValueError("Invalid view format. Supported formats are 'CHW' and 'HWC'.")

def discretize_mask(mask, threshold=0.5):
    t_mask = np.zeros(mask.shape)
    np.place(t_mask[:, :, 0], mask[:, :, 0] >= threshold, 1)
    return t_mask

def load_img(PATH: str, 
            size: Union[str, Tuple[int, int]] = 'original', 
            between: Tuple[float, float] = (0, 255),
            type: str = '' , view: str = ''):
    if type == 'numpy':
        img = np.load(PATH)
    else:
        img = plt.imread(PATH)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    
    img = resize_image(img=img, size=size)
    img = normalize_image(img=img, between=between)
    img = change_view(img, view)
    
    return img