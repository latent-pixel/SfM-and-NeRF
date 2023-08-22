import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import random
import argparse
import os
import numpy as np

from utils.data_utils import FetchImageData
from utils.network import MLP
from utils.nerf import nerf
from utils.misc_utils import tic, toc, calculate_psnr


def trainOperation(data_path, check_point_path, logs_path, near_plane, far_plane,
                   num_samples_per_ray, num_enc_freq_x, num_enc_freq_d, num_chunks,
                   num_epochs, learning_rate, latest_file, reshape_size, device):
    # Initialize the model
    x_posenc_shape, d_posenc_shape = 3*2*num_enc_freq_x + 3, 3*2*num_enc_freq_d + 3
    model = MLP(x_posenc_shape, d_posenc_shape, width=128).to(device)

    # Loading training and validation datasets 
    train_data = FetchImageData(data_path, split='train')
    val_data = FetchImageData(data_path, split='val')

    optimizer = Adam(model.parameters(), learning_rate)
    loss_func = torch.nn.functional.mse_loss

    # Tensorboard
    # Create a summary to monitor loss tensor
    writer = SummaryWriter(logs_path)
    if latest_file is not None:
        check_point = torch.load(check_point_path + latest_file + '.ckpt')
        # Extract only numbers from the name
        start_epoch = int(''.join(c for c in latest_file.split('a')[0] if c.isdigit()))
        model.load_state_dict(check_point['model_state_dict'])
        print('\nLoaded latest checkpoint with the name ' + latest_file + '....\n')
    else:
        start_epoch = 0
        print('\nNew model initialized....\n')
    
    start_timer = tic()
    depth_maps = []
    radiance_maps = []
    for epoch in range(start_epoch, num_epochs):
        print("Epoch [{}]".format(epoch+1))
        model.train()
        total_train_loss = 0.
        random.seed(epoch)
        img_idx = list(range(len(train_data)))
        random.shuffle(img_idx)
        pbar_inner = tqdm(total=len(train_data))
        for idx in img_idx:
            image = train_data.get_image(idx, reshape_size)
            cam_tfrm = train_data.get_camera_transforms(idx)
            img_height, img_width, _ = image.shape 
            K = cam_tfrm['intrinsic']
            Rt = cam_tfrm['extrinsic']
            F = K['fx']
            
            # Predict output/loss with forward pass
            nerf_func = nerf(img_height, img_width, F, Rt, near_plane, far_plane, model, num_samples_per_ray, num_enc_freq_x, num_enc_freq_d, num_chunks, device)
            train_loss = loss_func(nerf_func['rgb'], image.to(device))
            
            total_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar_inner.update()
        pbar_inner.close()
        avg_train_loss = total_train_loss / len(train_data)
        
        model.eval()
        total_val_loss = 0.
        psnr = 0.
        with torch.no_grad():
            for img_idx in range(len(val_data)):
                image = val_data.get_image(img_idx, reshape_size)
                cam_tfrm = val_data.get_camera_transforms(img_idx)
                img_height, img_width, _ = image.shape 
                K = cam_tfrm['intrinsic']
                Rt = cam_tfrm['extrinsic']
                F = K['fx']
                nerf_func = nerf(img_height, img_width, F, Rt, near_plane, far_plane, model, num_samples_per_ray, num_enc_freq_x, num_enc_freq_d, num_chunks, device)
                loss = loss_func(nerf_func['rgb'], image.to(device))
                total_val_loss += loss.item()
                psnr += calculate_psnr(loss)
        avg_val_loss = total_val_loss / len(val_data) # calculate loss and psnr avgs
        psnr /= len(val_data)
        print('avg_train_loss:{}, avg_val_loss:{}, avg_psnr:{}'.format(round(avg_train_loss, 4), round(avg_val_loss, 4), round(psnr, 4)))

        with torch.no_grad():
            image = train_data.get_image(19, reshape_size)
            cam_tfrm = train_data.get_camera_transforms(19)
            img_height, img_width, _ = image.shape 
            K = cam_tfrm['intrinsic']
            Rt = cam_tfrm['extrinsic']
            F = K['fx']
            nerf_func = nerf(img_height, img_width, F, Rt, near_plane, far_plane, model, num_samples_per_ray, num_enc_freq_x, num_enc_freq_d, num_chunks, device)
            # converting arrays to "images"
            depth_image = nerf_func['depth'].detach().cpu().numpy()
            depth_image = np.maximum(np.minimum(depth_image, np.ones_like(depth_image)), np.zeros_like(depth_image))
            depth_maps.append(depth_image)
            syn_image = nerf_func['rgb'].detach().cpu().numpy()
            syn_image = np.maximum(np.minimum(syn_image, np.ones_like(syn_image)), np.zeros_like(syn_image))
            radiance_maps.append(syn_image)
            
        # Update Tensorboard
        writer.add_scalar(f'Loss/TrainLoss', avg_train_loss, epoch)
        writer.add_scalar(f'Loss/ValLoss', avg_val_loss, epoch)
        writer.add_scalar('PSNR', psnr, epoch)
        writer.flush()  # without flushing, the tensorboard doesn't get updated until a lot of iterations!

        # Save model every epoch
        if (epoch) % 5 == 0:
            save_name = check_point_path + 'ep' + str(epoch) + '_model.ckpt'
            torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': train_loss}, save_name)
            print('Model saved at ' + save_name + '\n')

    # Saving the training images
    np.save('depth.npy', np.array(depth_maps))
    np.save('radiance.npy', np.array(radiance_maps))

    training_time = toc(start_timer)
    print("The total time taken to train the model: {} seconds".format(round(training_time, 2)))


def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the training process
    """
    # Parse Command Line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataPath', default='./phase2/data/lego/', help='Path to the dataset, Default: phase2/CIFAR10/')
    parser.add_argument('--CheckPointPath', default='./phase2/checkpoints/', help='Path to save Checkpoints, Default: phase2/checkpoints/')
    parser.add_argument('--LogsPath', default='./phase2/logs/', help='Path to save Logs for Tensorboard, Default=phase2/logs/')
    parser.add_argument('--NearPlane', type=float, default=2.0, help='Distance to the nearest plane of the scene from the camera')
    parser.add_argument('--FarPlane', type=float, default=6.0, help='Distance to the farthest plane of the scene from the camera')
    parser.add_argument('--NumSamples', type=int, default=64, help='Number of sample points from each ray')
    parser.add_argument('--EncodingFreqsX', type=int, default=10, help='Number of encoding frequencies for location x')
    parser.add_argument('--EncodingFreqsD', type=int, default=4, help='Number of encoding frequencies for direction d')
    parser.add_argument('--ChunkSize', type=int, default=5000, help='Split large ray bundles into chunks before feeding to the MLP')
    parser.add_argument('--NumEpochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--LearningRate', type=int, default=5e-4, help='Learning rate')
    parser.add_argument('--LatestFile', default=None, help='Load Model from latest Checkpoint from CheckPointsPath?')

    args = parser.parse_args()
    DataPath = args.DataPath
    CheckPointPath = args.CheckPointPath
    LogsPath = args.LogsPath
    NearPlane = args.NearPlane
    FarPlane = args.FarPlane
    NumSamples = args.NumSamples
    EncodingFreqsX = args.EncodingFreqsX
    EncodingFreqsD = args.EncodingFreqsD
    ChunkSize = args.ChunkSize
    NumEpochs = args.NumEpochs
    LearningRate = args.LearningRate
    LatestFile = args.LatestFile

    RESHAPE_SIZE = 100
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    NumChunks = RESHAPE_SIZE * RESHAPE_SIZE * NumSamples // ChunkSize

    if not os.path.exists(CheckPointPath):
        os.makedirs(CheckPointPath)

    trainOperation(DataPath, CheckPointPath, LogsPath, NearPlane, FarPlane, 
                   NumSamples, EncodingFreqsX, EncodingFreqsD, NumChunks,
                   NumEpochs, LearningRate, LatestFile, RESHAPE_SIZE, DEVICE)


if __name__ == '__main__':
    main()
