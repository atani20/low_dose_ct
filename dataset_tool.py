import os
import numpy as np
import h5py
import config
from pydicom import dcmread
from progress.bar import IncrementalBar


def get_patient_pixels(patient_path):
    # load list of slices and extract pixel array
    assert os.path.isdir(patient_path)
    slices = [dcmread(os.path.join(patient_path, s)) for s in os.listdir(patient_path)]
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    try:
        padding = slices[0].PixelPaddingValue
    except:
        padding = 0
    image[image == padding] = 0
    return np.array(image, dtype=np.int16)


def extract_patches(full_images, low_images, patch_size, alpha=0.5, patient_shuffle=True, patch_shuffle=True):
    assert 0 <= alpha <= 1
    assert full_images.shape == low_images.shape
    # cut images into pathes
    images_num, size, size1 = low_images.shape
    assert size >= patch_size
    full_out = np.empty((0, patch_size, patch_size), dtype='int16')
    low_out = np.empty((0, patch_size, patch_size), dtype='int16')
    images_idxs = np.random.permutation(np.arange(images_num)) if patient_shuffle else np.arange(images_num)
    n = size // patch_size
    k = 0
    for idx in images_idxs:
        full_img = full_images[idx, :, :]
        low_img = low_images[idx, :, :]
        full_patches = np.empty((0, patch_size, patch_size), dtype='int16')
        low_patches = np.empty((0, patch_size, patch_size), dtype='int16')

        patch_idxs1 = np.random.permutation(np.arange(n)) if patch_shuffle else np.arange(n)
        patch_idxs2 = np.random.permutation(np.arange(n)) if patch_shuffle else np.arange(n)
        # cut one image into pathes
        for i in patch_idxs1:
            for j in patch_idxs2:
                full_patch = full_img[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size]
                # if pach is black (all pixels less than 250) than add it to dataset with probability alpha
                if np.all(full_patch <= 250):
                    if np.random.uniform(0, 1) > alpha:
                        k += 1
                        continue
                full_patches = np.concatenate((full_patches, np.reshape(full_patch, (1,) + full_patch.shape)))

                low_patch = low_img[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size]
                low_patches = np.concatenate((low_patches, np.reshape(low_patch, (1,) + low_patch.shape)))
                
        full_out = np.concatenate((full_out, full_patches))
        low_out = np.concatenate((low_out, low_patches))
    return full_out, low_out


def dataset_prerprocessing(path, patch_size):
    assert os.path.isdir(path)
    patient_list = os.listdir(path)
    bar = IncrementalBar('Patient', max=len(patient_list))
    for patient in patient_list:
        bar.next()
        doses = os.listdir(os.path.join(path, patient))
        full_pixels = get_patient_pixels(os.path.join(path, patient, doses[0]))
        low_pixels = get_patient_pixels(os.path.join(path, patient, doses[1]))
        full_patches, low_patches = extract_patches(full_pixels, low_pixels, patch_size)
        filename = os.path.join(config.preproc_data, patient + '.h5py')
        with h5py.File(filename, 'a') as f:
            f.create_dataset('low', data=full_patches)
            f.create_dataset('full', data=low_patches)
        del full_patches
        del low_patches
    bar.finish()


if __name__ == "__main__":
    dataset_prerprocessing(config.data_dir, config.patch_size)