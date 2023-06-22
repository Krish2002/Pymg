import numpy as np
import matplotlib.pyplot as plt
import cv2 as op
from typing import List, Tuple, Union

        
def resize_image(
            img: List[float], 
            size: Union[str, Tuple[int, int]]):
    
    width, height = img.shape[0], img.shape[1]
    
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


def load_img(PATH: str, 
            size: Union[str, Tuple[int, int]] = 'original', 
            between: Tuple[float, float] = (0, 255)):

    img = plt.imread(PATH)
    
    img = resize_image(img=img, size=size)
    img = normalize_image(img=img, between=between)
    
    return img