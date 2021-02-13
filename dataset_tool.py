import os
from pydicom import dcmread
import numpy as np
import h5py
import config
import random


def get_patient_pixels(patient_path):
    slices = [dcmread(os.path.join(patient_path, s)) for s in os.listdir(patient_path)]
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    try:
        padding = slices[0].PixelPaddingValue
    except:
        padding = 0
    image[image == padding] = 0
    return np.array(image, dtype=np.int16)


def extract_patches(nd_image, ld_image, patch_size, stride=32, skip_images=30):
    images_num, h, w = ld_image.shape
    images_num //= skip_images
    out_ld = np.empty((0, patch_size, patch_size))
    out_nd = np.empty((0, patch_size, patch_size))
    sz = ld_image.itemsize
    shape = ((h - patch_size) // stride + 1, (w - patch_size) // stride + 1, patch_size, patch_size)
    strides = sz * np.array([w * stride, stride, w, 1])
    images_idxs = random.sample(range(1, ld_image.shape[0]), images_num)
    for d, idx in enumerate(images_idxs):
        ld_patches = np.lib.stride_tricks.as_strided(ld_image[idx, :, :], shape=shape, strides=strides)
        ld_blocks = ld_patches.reshape(-1, patch_size, patch_size)
        out_ld = np.concatenate((out_ld, ld_blocks[:, :, :]))
        nd_patches = np.lib.stride_tricks.as_strided(nd_image[idx, :, :], shape=shape, strides=strides)
        nd_blocks = nd_patches.reshape(-1, patch_size, patch_size)
        out_nd = np.concatenate((out_nd, nd_blocks[:, :, :]))
    return out_nd[:, :, :],out_ld[:, :, :]


# создание датасета, режем на кусочки размера patch_size
def preproc_patient(path, patch_size):
    full_patches = np.empty((0, patch_size, patch_size))
    low_patches = np.empty((0, patch_size, patch_size))
    for patient in os.listdir(path)[0:10]:
        doses = os.listdir(os.path.join(path, patient))
        full_pixels = get_patient_pixels(os.path.join(path, patient, doses[0]))
        low_pixels = get_patient_pixels(os.path.join(path, patient, doses[1]))
        full, low = extract_patches(full_pixels, low_pixels, patch_size)
        full_patches = np.concatenate((full_patches, full))
        low_patches = np.concatenate((low_patches, low))
        print(patient)
    print(full_patches.shape)
    return full_patches, low_patches


if __name__ == "__main__":
    full, low = preproc_patient(config.data_dir, config.patch_size)
    filename = os.path.join(config.preproc_data_dir, 'ldct_10_p.h5')
    with h5py.File(filename, 'a') as f:
        f.create_dataset('low', data=low) 
        f.create_dataset('full', data=full)