import torch
from utils.data_utils import FetchImageData

image_data = FetchImageData('phase2/data/lego/', split='train')

image_data.visualize_cam_poses()