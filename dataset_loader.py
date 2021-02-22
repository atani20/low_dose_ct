from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import h5py
import os


class LowCTDataset(Dataset):
    def __init__(self, path, transforms=None):
        h5_file = h5py.File(path, 'r')
        self.low_data = h5_file.get('low')
        self.full_data = h5_file.get('full')
        assert self.low_data.shape == self.full_data.shape
        self.size = self.low_data.shape[0]
        self.transforms = transforms

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x = self.low_data[index] / 4095
        y = self.full_data[index] / 4095
        x = x.reshape((1,) + x.shape)
        y = y.reshape((1,) + y.shape)
        if self.transforms:
            x = self.transforms(x)
            y = self.transforms(y)
        return x, y


def get_data_loader(dataset_path, batch_size=32, inx_from=0, indx_to=45, num_workers=0, transforms=None):
    dataset_files = glob.glob(os.path.join(os.path.abspath(dataset_path), "*.h5py"))[inx_from:indx_to]
    datasets = []
    for file in dataset_files:
        datasets.append(LowCTDataset(os.path.join(dataset_path, file), transforms))
    data = ConcatDataset(datasets)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return data_loader
