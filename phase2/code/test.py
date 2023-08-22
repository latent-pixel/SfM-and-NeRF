import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.data_utils import FetchImageData
from utils.network import MLP
from utils.nerf import nerf
from utils.misc_utils import calculate_psnr, arr_to_gif


def testOperation(data_path, reshape_size, near_plane, far_plane, 
                  num_samples_per_ray, num_enc_freq_x, num_enc_freq_d, 
                  num_chunks, model_path, results_path, device):
    
    # Initialize the model
    x_posenc_shape, d_posenc_shape = 3*2*num_enc_freq_x + 3, 3*2*num_enc_freq_d + 3
    model = MLP(x_posenc_shape, d_posenc_shape, width=128).to(device)

    # Loading training and validation datasets 
    test_data = FetchImageData(data_path, split='test')

    if(not(os.path.isfile(model_path))):
        print('ERROR: Model does not exist in ' + model_path)
        sys.exit()
    else:
        CheckPoint = torch.load(model_path)

    model.load_state_dict(CheckPoint['model_state_dict'])
    print('Model loaded...\n')
    print('Number of parameter groups in this model: {}'.format(len(model.state_dict().items())))
    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.eval()
    
    loss_func = torch.nn.functional.mse_loss

    images_list = []
    total_loss = 0.
    psnr = 0.
    with torch.no_grad():
        pbar_inner = tqdm(total=len(test_data))
        for img_idx in range(len(test_data)):
            image = test_data.get_image(img_idx, reshape_size)
            cam_tfrm = test_data.get_camera_transforms(img_idx)
            img_height, img_width, _ = image.shape 
            K = cam_tfrm['intrinsic']
            Rt = cam_tfrm['extrinsic']
            F = K['fx']
            
            nerf_func = nerf(img_height, img_width, F, Rt, near_plane, far_plane, model, num_samples_per_ray, num_enc_freq_x, num_enc_freq_d, num_chunks, device)
            loss = loss_func(nerf_func['rgb'], image.to(device))
            total_loss += loss.item()
            psnr += calculate_psnr(loss)
            
            syn_image = nerf_func['rgb'].detach().cpu().numpy()
            syn_image = np.maximum(np.minimum(syn_image, np.ones_like(syn_image)), np.zeros_like(syn_image))
            images_list.append(syn_image)
            # plt.imsave(results_path + 'test_img{}.png'.format(img_idx), syn_image)
            
            pbar_inner.update()
        pbar_inner.close()
    
    loss = total_loss / len(test_data)
    psnr /= len(test_data)
    print('avg_loss:{}, avg_psnr:{}'.format(round(loss, 4), round(psnr, 4)))
    
    # saving the images
    images_list = np.array(images_list)
    arr_to_gif(images_list, results_path)


DATA_PATH = 'phase2/data/lego/'
MODEL_PATH = 'phase2/checkpoints/ep195_model.ckpt'
RESULTS_PATH = 'phase2/results/'

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

RESHAPE_SIZE = 100
Z_N, Z_F = 2.0, 6.0
N_SAMPLES = 64
L_X, L_D = 10, 4
CHUNK_SIZE = 2048
NUM_CHUNKS = RESHAPE_SIZE * RESHAPE_SIZE * N_SAMPLES // CHUNK_SIZE

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    testOperation(DATA_PATH, RESHAPE_SIZE, Z_N, Z_F, 
                  N_SAMPLES, L_X, L_D, 
                  NUM_CHUNKS, MODEL_PATH, RESULTS_PATH, DEVICE)
    