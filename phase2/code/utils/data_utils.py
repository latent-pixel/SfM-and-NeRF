import json
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

        self._camera_angle_x = self.img_data['camera_angle_x']
        self._frames = self.img_data['frames']
    

    def __len__(self):
        return len(self._frames)


    def get_camera_angle(self):
        """
        Returns:
            torch.FloatTensor: The FOV in x dimension
        """
        return torch.tensor(self._camera_angle_x)
    
    
    def get_image(self, idx):
        if idx not in range(len(self._frames)):
            raise ValueError('Index out of bounds.')
        img_path = self.base_path + self._frames[idx]['file_path'] + '.png'
        og_image = np.array(Image.open(img_path)).astype(np.float32) / 255.
        r, g, b, a = og_image[:, :, 0], og_image[:, :, 1], og_image[:, :, 2], og_image[:, :, 3] 
        # rgba -> rgb conversion, source: https://stackoverflow.com/questions/2049230/convert-rgba-color-to-rgb
        image = np.zeros(shape=(og_image.shape[0], og_image.shape[1], 3), dtype=np.float32)
        image[:, :, 0] = (1. - a) * 1. + a * r
        image[:, :, 1] = (1. - a) * 1. + a * g
        image[:, :, 2] = (1. - a) * 1. + a * b
        return torch.from_numpy(image)
    

    def get_camera_transforms(self, idx):
        if idx not in range(len(self._frames)):
            raise ValueError('Index out of bounds.')
        camera_extrinsic = self._frames[idx]['transform_matrix']
        img_path = self.base_path + self._frames[idx]['file_path'] + '.png'
        og_image = np.array(Image.open(img_path))
        f = 0.5 * og_image.shape[1] / np.tan(0.5 * self._camera_angle_x)
        camera_intrinsic = {
            'fx' : f,
            'fy' : f,
            'cx' : og_image.shape[1] / 2,
            'cy' : og_image.shape[0] / 2
        }
        camera_extrinsic = torch.from_numpy(np.array(camera_extrinsic, dtype=np.float32))
        return {'intrinsic' : camera_intrinsic, 'extrinsic' : camera_extrinsic}
    

    def visualize_cam_poses(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for idx in range(len(self._frames)):
            extrinsic = np.array(self._frames[idx]['transform_matrix'])
            R = extrinsic[:3, :3]
            t = extrinsic[:3, -1]
            direction_vec = np.array([0, -0, -1], dtype=np.float32)
            direction_vec = np.expand_dims(direction_vec, axis=0)
            world_direction_vec = direction_vec * R
            world_direction_vec = np.sum(world_direction_vec, axis=-1)
            x, y, z = t
            u, v, w = world_direction_vec
            ax.quiver(x, y, z, u, v, w, length=0.35, normalize=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()