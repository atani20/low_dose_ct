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


def save_current_result(save_path, network, iter, train_losses, test_loss_x, test_loss_y):
    zeros = 15
    iter_str = str(iter).zfill(zeros - len(str(iter)))
    # for kaggle only

    # def save_to_google(name, path, fid='1C4l9TsB0Hh1ZjaWDy4iTQKMy6PnxWZUM'):
    #     file = drive.CreateFile({'title': name, "parents": [{'id': fid}]})
    #     file.SetContentFile(path)
    #     file.Upload()
    name = 'network_' + iter_str + '.pkl'
    path = os.path.join(save_path, name)
    torch.save(network.state_dict(), path)
    # save_to_google(name, path)

    name = 'loss_' + iter_str + '.npy'
    path = os.path.join(save_path, name)
    np.save(path, np.array(train_losses))
    # save_to_google(name, path)

    name = 'test_loss_x' + iter_str + '.npy'
    path = os.path.join(save_path, name)
    np.save(path, np.array(test_loss_x))
    # save_to_google(name, path)

    name = 'test_loss_y' + iter_str + '.npy'
    path = os.path.join(save_path, name)
    np.save(path, np.array(test_loss_y))
    # save_to_google(name, path)


def get_test_loss(network, data_loader, device):
    results = []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            pred = network(x)
            size = x.size()[2]

            # y = y.detach().cpu().numpy().reshape((size, -1))
            # pred = pred.detach().cpu().numpy().reshape((size, -1))
            results.append(metrics.get_mse(pred, y))
    return sum(results) / len(data_loader)

def train(network,
          data_loader,
          optimizer,
          criterion,
          lr,
          device,
          save_path,
          decay_iters=500,
          num_epochs=100,
          print_iter=100,
          network_snapshot_iter=100,
          iter_to_test=1000,
          data_loader_test=None):
    train_losses = []
    total_iters = 0
    test_loss_x = []
    test_loss_y = []
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
            # test
            if total_iters % iter_to_test == 0 and data_loader_test is not None:
                res = get_test_loss(network, data_loader_test, device)
                test_loss_x.append(total_iters)
                test_loss_y.append(res)

            # save
            if total_iters % network_snapshot_iter == 0:
                save_current_result(save_path, network, total_iters, train_losses)

            # print
            if total_iters % print_iter == 0:
                print('total_iters %-4d epoch %-4d loss %-0.6f total_time ' % (
                        total_iters, epoch, loss.item()),  format_time(time.time() - start_time))

            # learning rate decay
            if total_iters % decay_iters == 0:
                lr *= 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


def test_single(network, data_loader, device):
    # PSNR, SSIM, RMSE
    psnr_all, ssim_all, rmse_all = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.float().to(device)
            y = y.float().to(device)

            pred = network(x)
            predict_result = metrics.get_metrics(y, pred)

            psnr_all.append(predict_result[0])
            ssim_all.append(predict_result[1])
            rmse_all.append(predict_result[2])
        n = len(data_loader)
        return sum(psnr_all) / n, sum(ssim_all) / n, sum(rmse_all) / n


def test_all(directory, data_loader, device):
    files = glob.glob(os.path.join(directory, 'network_*.pkl'))
    with open('result.csv', 'w') as f:
        f.write(f'netw_num \t PSNR avg \t SSIM avg \tRMSE avg \n')
        print(f'netw_num \t PSNR avg \t SSIM avg \tRMSE avg \n')
        for file in files:
            netw_num = int(file.split('.')[-2][-7:])
            network.load_state_dict(torch.load(file))
            network.eval()
            predict = test_single(network, data_loader, device)

            f.write(f'{netw_num: 10d}\t{predict[0]:.6f}\t{predict[1]:.6f}\t{predict[2]:.6f}\n')
            print(f'{netw_num: 10d}\t{predict[0]:.6f}\t{predict[1]:.6f}\t{predict[2]:.6f}\n')


def get_network_pkl_path():
    snapshots = glob.glob(os.path.join(os.path.abspath(config.result_dir), "network_*.pkl"))
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

    data_loader_test = get_data_loader(os.path.abspath(config.preproc_data), inx_from=45, indx_to=50, batch_size=64)
    if config.mode == 'train':
        print('Start training...')
        data_loader_train = get_data_loader(os.path.abspath(config.preproc_data))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(network.parameters(), config.lr)
        train(network=network, data_loader=data_loader_train, optimizer=optimizer, criterion=criterion,
              lr=config.lr, device=device, save_path=config.result_dir, data_loader_test=data_loader_test)
        print('End training.')

    elif config.mode == 'test':
        print('Start test...')
        test_all('C:/Users/ACER/Desktop/диплом/low_dose_ct/result', data_loader_test, device)
        print('End test.')
    else:
        print('Unknown mode:', config.mode)