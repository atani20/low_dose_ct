import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def get_metrics(y_real, y_pred):
    y_real, y_pred = prepare_data(y_real, y_pred)
    psnr = get_psnr(y_real, y_pred, False)
    ssim = get_ssim(y_pred, y_real, False)
    rmse = get_rmse(y_pred, y_real, False)
    return psnr, ssim, rmse


def prepare_data(y_real, y_pred):
    size = y_real.size()[2]
    y_real = y_real.detach().cpu().numpy().reshape((size, -1))
    y_pred = y_pred.detach().cpu().numpy().reshape((size, -1))
    y_pred = np.clip(y_pred, 0, 1)
    return y_real, y_pred


def get_psnr(y_real, y_pred, is_prepare=True):
    if is_prepare:
        y_pred, y_real = prepare_data(y_pred, y_real)
    return psnr(y_pred, y_real)


def get_ssim(y_real, y_pred, is_prepare=True):
    if is_prepare:
        y_pred, y_real = prepare_data(y_pred, y_real)
    return ssim(y_pred, y_real)


def get_rmse(y_real, y_pred, is_prepare=True):
    if is_prepare:
        y_pred, y_real = prepare_data(y_pred, y_real)
    return np.sqrt(mse(y_pred, y_real))

def get_mse(y_real, y_pred, is_prepare=True):
    if is_prepare:
        y_pred, y_real = prepare_data(y_pred, y_real)
    return mse(y_pred, y_real)