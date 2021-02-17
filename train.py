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


def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60:         return '%ds'                % (s)
    elif s < 60*60:    return '%dm %02ds'          % (s // 60, s % 60)
    elif s < 24*60*60: return '%dh %02dm %02ds'    % (s // (60*60), (s // 60) % 60, s % 60)
    else:              return '%dd %02dh %02dm'    % (s // (24*60*60), (s // (60*60)) % 24, (s // 60) % 60)


def save_model(save_path, network, iter):
    f = os.path.join(save_path, f'redcnn_{iter}.pkl')
    torch.save(network.state_dict(), f)


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
          image_snapshot_iter=2000,
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
                save_model(save_path, network, total_iters)
                np.save(os.path.join(save_path, f'loss_{total_iters}.npy'), np.array(train_losses))

            # print and pictures
            if total_iters % print_iter == 0:
                print('total_iters %-4d epoch %-4d loss %-0.6f total_time ' % (
                        total_iters, epoch, loss.item()),  format_time(time.time() - start_time))

            # TODO: pictures saving

            # learning rate decay
            if total_iters % decay_iters == 0:
                lr *= 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


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
    data_loader_train = get_data_loader(os.path.abspath(config.preproc_data))
    lr = config.lr
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr)
    train(network=network, data_loader=data_loader_train, optimizer=optimizer, criterion=criterion,
          lr=lr, device=device, save_path=config.result_dir, )

