import time
import torch
from PIL import Image
import os
import numpy as np


def tic():
    """
    Function to start timer
    Tries to mimic tic() toc() in MATLAB
    """
    StartTime = time.time()
    return StartTime


def toc(StartTime):
    """
    Function to stop timer
    Tries to mimic tic() toc() in MATLAB
    """
    return time.time() - StartTime


def calculate_psnr(mse):
    return (-10.0 * torch.log10(mse)).item()   


def upscale_images(images_path, final_size=600):
    for idx, filename in enumerate(sorted(os.listdir(images_path))):
        if ".png" in filename:
            image = Image.open(os.path.join(images_path, filename))
            resized_image = image.resize((final_size, final_size), resample=Image.BICUBIC)
            resized_image.save(images_path+"resized_img{}.png".format(idx))     


def img_to_gif(npy_file_path):
    images_array = np.load(npy_file_path)
    images_array = (images_array * 255).astype(np.uint8)
    image_frames = []
    for image in images_array:
        image = Image.fromarray(image)
        image_frames.append(image)
    image_frames[0].save('output.gif', save_all=True, append_images=image_frames[1:], loop=0, duration=100)


def arr_to_gif(images_array, save_path):
    images_array = (images_array * 255).astype(np.uint8)
    image_frames = []
    for image in images_array:
        image = Image.fromarray(image)
        image_frames.append(image)
    image_frames[0].save(save_path+'output.gif', save_all=True, append_images=image_frames[1:], loop=0, duration=100)
