import pymg as pymg
import matplotlib.pyplot as plt

def main():
    
    PATH = '../coldplay.jpg'
    img = pymg.load_img(PATH, between=(0, 5))
    print(img.shape)
    
    img = pymg.resize_image(img, 'half')
    print(img.shape)

if __name__ == "__main__":
    main()