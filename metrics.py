import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def get_metrics(x, y, pred):
    size = x.size()[2]
    x = x.detach().cpu().numpy().reshape((size, -1))
    y = y.detach().cpu().numpy().reshape((size, -1))
    original = (psnr(x, y), ssim(x, y), np.sqrt(mse(x, y)))
    predict = (psnr(pred, y), ssim(pred, y), np.sqrt(mse(pred, y)))
    return original, predict