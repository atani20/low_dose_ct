from dataset_loader import get_data_loader
from red_cnn import RED_CNN

import os
import glob
import config
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import metrics
import h5py

def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60:         return '%ds'                % (s)
    elif s < 60*60:    return '%dm %02ds'          % (s // 60, s % 60)
    elif s < 24*60*60: return '%dh %02dm %02ds'    % (s // (60*60), (s // 60) % 60, s % 60)
    else:              return '%dd %02dh %02dm'    % (s // (24*60*60), (s // (60*60)) % 24, (s // 60) % 60)


def save_current_result(save_path, network, iter, train_losses):
    f = os.path.join(save_path, f'network_{iter}.pkl')
    torch.save(network.state_dict(), f)
    np.save(os.path.join(save_path, f'loss_{iter}.npy'), np.array(train_losses))


def train(network,
          data_loader,
          optimizer,
          criterion,
          lr,
          device,
          save_path,
          decay_iters=3000,
          num_epochs=1000,
          print_iter=20,
          network_snapshot_iter=100):
    train_losses = []
    total_iters = 0
    start_time = time.time()
    for epoch in range(1, num_epochs):
        network.train(True)
        for i, (x, y) in enumerate(data_loader):
            total_iters += 1

            x = x.float().to(device)
            y = y.float().to(device)

            pred = network(x)
            loss = criterion(pred, y)
            network.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # save
            if total_iters % network_snapshot_iter == 0:
                save_current_result(save_path, network, total_iters, train_losses)

            # print and pictures
            if total_iters % print_iter == 0:
                print('total_iters %-4d epoch %-4d loss %-0.6f total_time ' % (
                        total_iters, epoch, loss.item()),  format_time(time.time() - start_time))

            # learning rate decay
            if total_iters % decay_iters == 0:
                lr *= 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

def save_metrics(original, predict, dir_to_save):
    filename = os.path.join(dir_to_save, 'all_metrics.hp5y')
    with h5py.File(filename, 'a') as f:
        f.create_dataset('original_psnr', data=original[0])
        f.create_dataset('original_ssim', data=original[1])
        f.create_dataset('original_rmse', data=original[2])

        f.create_dataset('predict_psnr', data=predict[0])
        f.create_dataset('predict_ssim', data=predict[1])
        f.create_dataset('predict_rmse', data=predict[2])

    n = len(original[0])
    filename = os.path.join(dir_to_save, 'metrics.txt')
    with open(filename, 'w') as f:
        f.write(f'Original: \n PSNR avg: {sum(original[0]) / n :5.f} \n '
                f'SSIM avg: {sum(original[1]) / n:.5f} \nRMSE avg: {sum(original[2]) / n:.5f}\n\n')

        f.write(f'Predict: \n PSNR avg: {sum(predict[0]) / n :5.f} \n '
                f'SSIM avg: {sum(predict[1]) / n:.5f} \nRMSE avg: {sum(predict[2]) / n:.5f}\n\n')

def test(network, data_loader, device, dir_to_save):
    # PSNR, SSIM, RMSE
    original_psnr, original_ssim, original_rmse = [], [], []
    predict_psnr, predict_ssim, predict_rmse = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            pred = network(x)

            original_result, predict_result = metrics.get_metrics(x, y, pred)
            original_psnr.append(original_result[0])
            original_ssim.append(original_result[1])
            original_rmse.append(original_result[2])

            predict_psnr.append(predict_result[0])
            predict_ssim.append(predict_result[1])
            predict_rmse.append(predict_result[2])
        original = (original_psnr, original_ssim, original_rmse)
        predict = (predict_psnr, predict_ssim, predict_rmse)
        save_metrics(original, predict, dir_to_save)
        n = len(data_loader)




def get_network_pkl_path():
    snapshots = glob.glob(os.path.join(os.path.abspath(config.result_dir), "redcnn_*.pkl"))
    if len(snapshots) == 0:
        return None
    return snapshots[-1]


if __name__ == "__main__":
    snap_path = get_network_pkl_path()
    network = RED_CNN()
    if snap_path is not None:
        print('Loading networks from "%s"...' % snap_path)
        network.load_state_dict(torch.load(snap_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)

    if config.mode == 'train':
        data_loader_train = get_data_loader(os.path.abspath(config.preproc_data))
        lr = config.lr
        criterion = nn.MSELoss()
        optimizer = optim.Adam(network.parameters(), lr)
        train(network=network, data_loader=data_loader_train, optimizer=optimizer, criterion=criterion,
            lr=lr, device=device, save_path=config.result_dir)

    elif config.mode == 'test':
        data_loader_test = get_data_loader(os.path.abspath(config.preproc_data), inx_from=45, indx_to=50, batch_size=1)
        test(network, data_loader_test, device, dir_to_save=config.result_dir)
    else:
        print('Unknown mode:', config.mode)