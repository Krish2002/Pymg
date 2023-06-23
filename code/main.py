import pymg as pymg
import matplotlib.pyplot as plt

def main():
    
    PATH = '../coldplay.jpg'
    PATH_Mask = '../mask.png'
    img = pymg.load_img(PATH_Mask, between=(0, 1) , view='HWC' , size='original')
    print(img.shape)
    
    img  = pymg.resize_image(img, size=(125 , 125))
    img = pymg.discretize_mask(img, threshold=0.5)
    img = pymg.change_view(img , 'CHW')
    print(img.shape)

if __name__ == "__main__":
    main()