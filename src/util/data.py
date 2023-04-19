from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from constants import CLEAN_TRAIN_IMG_PATH, CLEAN_TRAIN_MSK_PATH, CLEAN_VAL_MSK_PATH, CLEAN_VAL_IMG_PATH


class SimpleLogger:

    def __init__(self, debug=True):
        self.debug = debug

    def enable_debug(self):
        self.debug = True

    def disable_debug(self):
        self.debug = False

    def log(self, message, condition=True):
        if self.debug and condition:
            print(message)


logger = SimpleLogger(debug=True)


def to_categorical(y, n_classes):
    return np.eye(n_classes, dtype="uint8")[y]


class BraTSDataset(Dataset):

    def log(self, message):
        logger.log(message, condition=self.debug)

    def __init__(self, images_path, masks_path, transform=None, one_hot_target=True, debug=True):
        self.images = sorted(glob(f"{images_path}/*.npy"))
        self.masks = sorted(glob(f"{masks_path}/*.npy"))
        self.transform = transform
        self.one_hot_target = one_hot_target
        self.debug = debug
        self.log(f"images: {len(self.images)}, masks: {len(self.masks)} ")
        assert len(self.images) == len(self.masks), "images and masks lengths are not the same!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        image = np.load(self.images[idx])
        mask = np.load(self.masks[idx])

        if self.one_hot_target:
            mask = to_categorical(mask, 4)

        image = torch.from_numpy(image).float()  # .double()
        mask = torch.from_numpy(mask)  # .float() #.long()

        return image.permute((3, 0, 1, 2)), mask.permute((3, 0, 1, 2))


def get_train_ds():
    return BraTSDataset(CLEAN_TRAIN_IMG_PATH, CLEAN_TRAIN_MSK_PATH)


def get_val_ds():
    return BraTSDataset(CLEAN_VAL_IMG_PATH, CLEAN_VAL_MSK_PATH)


def get_dl(dataset, batch_size=32, pm=True, nw=4):
    return DataLoader(dataset, batch_size, shuffle=True, pin_memory=pm, num_workers=nw,)


# this is for testing only
if __name__ == '__main__':
    train_ds = BraTSDataset(CLEAN_TRAIN_IMG_PATH, CLEAN_TRAIN_MSK_PATH)
    print(train_ds[0][0].shape)
    print(train_ds[0][1].shape)
    dl = get_dl(train_ds, batch_size=1)
    print("OK")
