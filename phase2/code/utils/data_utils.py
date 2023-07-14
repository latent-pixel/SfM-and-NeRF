import json
from PIL import Image
import numpy as np
import torch


class FetchImageData():
    """Loads image data: the required data can be obtained using 
    methods in the class
    """
    def __init__(self, base_path, split='train') -> None:
        """
        Args:
            base_path (string): Path to the data directory
            split (str, optional): Dataset to load - train, validate or test. Defaults to 'train'.
        """
        self.base_path = base_path

        if split == 'train':
            file = open(base_path+'transforms_train.json')
            self.img_data = json.load(file)
            file.close()
        elif split == 'val':
            file = open(base_path+'transforms_val.json')
            self.img_data = json.load(file)
            file.close()
        elif split == 'test':
            file = open(base_path+'transforms_test.json')
            self.img_data = json.load(file)
            file.close()
        else:
            raise ValueError('Could not find data for the specified base_path/split!')

        self.camera_angle_x = self.img_data['camera_angle_x']
        self.frames = self.img_data['frames']
    

    def get_camera_angle(self):
        return self.camera_angle_x
    
    
    def get_transform(self, idx):
        if idx not in range(len(self.frames)):
            raise ValueError('Index out of bounds.')
        return self.frames[idx]['transform_matrix']
    
    
    def get_image(self, idx):
        if idx not in range(len(self.frames)):
            raise ValueError('Index out of bounds.')
        img_path = self.base_path + self.frames[idx]['file_path'] + '.png'
        image = Image.open(img_path)
        return image
    