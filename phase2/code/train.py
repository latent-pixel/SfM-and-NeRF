import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

from utils.data_utils import FetchImageData
from utils.network import MLP
from utils.nerf import nerf
from utils.misc_utils import tic, toc, calculate_psnr


def TrainOperation(data_path, reshape_size, learning_rate, near_plane, far_plane, 
                   num_samples_per_ray, num_enc_freq_x, num_enc_freq_d, num_chunks, 
                   num_epochs, latest_file, logs_path, check_point_path, device):
    # Initialize the model
    x_posenc_shape, d_posenc_shape = 3*2*num_enc_freq_x + 3, 3*2*num_enc_freq_d + 3
    mlp = MLP(x_posenc_shape, d_posenc_shape, width=128).to(device)

    # Loading training and validation datasets 
    train_data = FetchImageData(data_path, split='train')
    val_data = FetchImageData(data_path, split='val')

    # num_train_samples = len(train_data)

    optimizer = Adam(mlp.parameters(), learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_func = torch.nn.functional.mse_loss

    # Tensorboard
    # Create a summary to monitor loss tensor
    writer = SummaryWriter(logs_path)
    if latest_file is not None:
        check_point = torch.load(check_point_path + latest_file + '.ckpt')
        # Extract only numbers from the name
        start_epoch = int(''.join(c for c in latest_file.split('a')[0] if c.isdigit()))
        mlp.load_state_dict(check_point['model_state_dict'])
        print('\nLoaded latest checkpoint with the name ' + latest_file + '....\n')
    else:
        start_epoch = 0
        print('\nNew model initialized....\n')
    
    start_timer = tic()
    for Epoch in range(start_epoch, num_epochs):
        print("Epoch [{}]".format(Epoch+1))
        # print("Learning rate: ", Optimizer.param_groups[0]['lr'])
        mlp.train()
        total_train_loss = 0.
        # epoch_history = []
        pbar_inner = tqdm(total=len(train_data))
        for img_idx in range(len(train_data)):
            image = train_data.get_image(img_idx, reshape_size)
            cam_tfrm = train_data.get_camera_transforms(img_idx)
            img_height, img_width, _ = image.shape 
            K = cam_tfrm['intrinsic']
            Rt = cam_tfrm['extrinsic']
            F = K['fx']

            # Predict output/loss with forward pass
            nerf_func = nerf(img_height, img_width, F, Rt, near_plane, far_plane, mlp, num_samples_per_ray, num_enc_freq_x, num_enc_freq_d, num_chunks, device)
            train_loss = loss_func(nerf_func['rgb'], image.to(device))
            total_train_loss += train_loss.item()

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar_inner.update()

        pbar_inner.close()
        avg_train_loss = total_train_loss / len(train_data)
        result = {'avg_train_loss': avg_train_loss}
        
        mlp.eval()
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
                nerf_func = nerf(img_height, img_width, F, Rt, near_plane, far_plane, mlp, num_samples_per_ray, num_enc_freq_x, num_enc_freq_d, num_chunks, device)
                loss = loss_func(nerf_func['rgb'], image.to(device))
                total_val_loss += loss.item()
                psnr += calculate_psnr(loss)

        avg_val_loss = total_val_loss / len(val_data) # calculate loss and psnr avgs
        psnr /= len(val_data)
        result.update({'avg_val_loss': avg_val_loss, 'avg_psnr': psnr})
        # epoch_history.append(result)
        print(result)

        # result = model.epoch_end(EpochHistory)  # calculates the loss and acc avgs
        # model.fetch_epoch_results(result)  # prints the epoch loss and accuracy

        # Update Tensorboard
        writer.add_scalar(f'Loss/TrainLoss', result['avg_train_loss'], Epoch)
        writer.add_scalar(f'Loss/ValLoss', result['avg_val_loss'], Epoch)
        writer.add_scalar('PSNR', result["avg_psnr"], Epoch)
        writer.flush()  # Without flushing, the tensorboard doesn't get updated until a lot of iterations!

        # Save model every epoch
        save_name = check_point_path + 'ep' + str(Epoch+1) + '_model.ckpt'
        torch.save({'epoch': Epoch,'model_state_dict': mlp.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': train_loss}, save_name)
        print('Model saved at ' + save_name + '\n')
        
        scheduler.step() # Decay learning rate at each subsequent "step"

    training_time = toc(start_timer)
    print("The total time taken to train the model: {} seconds".format(round(training_time, 2)))


DATA_PATH = 'phase2/data/lego/'
LOGS_PATH = 'phase2/logs'
CHECKPOINTS_PATH = 'phase2/checkpoints/'

RESHAPE_SIZE = 100
Z_N, Z_F = 2.0, 6.0
N_SAMPLES = 64
L_X, L_D = 10, 4
CHUNK_SIZE = 2048
NUM_CHUNKS = RESHAPE_SIZE * RESHAPE_SIZE * N_SAMPLES // CHUNK_SIZE

NUM_EPOCHS = 5
LEARNING_RATE = 5e-4

LATEST_FILE = None

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    TrainOperation(DATA_PATH, RESHAPE_SIZE, LEARNING_RATE, Z_N, Z_F, 
                   N_SAMPLES, L_X, L_D, NUM_CHUNKS, NUM_EPOCHS, 
                   LATEST_FILE, LOGS_PATH, CHECKPOINTS_PATH, DEVICE)

# image_data = FetchImageData('phase2/data/lego/', split='train')

# img_idx = 19
# image = image_data.get_image(img_idx)
# cam_tfrm = image_data.get_camera_transforms(img_idx)

# K = cam_tfrm['intrinsic']
# R_T = cam_tfrm['extrinsic']

# F = K['fx']

# x_posenc_shape, d_posenc_shape = 3*2*L_X, 3*2*L_D
# mlp = MLP(x_posenc_shape, d_posenc_shape, width=128).to(device)

# nerf_func = nerf(H, W, F, R_T, Z_N, Z_F, N_SAMPLES, L_X, L_D, NUM_CHUNKS, device)
