import os
import pandas as pd
import torch
# from torchvision.io import read_image
from torch.utils.data import Dataset
from skimage import io

# the __init__ purpose is to intialize annotation files,
# images and transformer that we want out data to
# go through
class CustomImageDataset(Dataset):
    def __init__(self, label_file,
                 img_dir, transform=None,
                 target_transform=None):
        self.img_labels = pd.read_csv(label_file)
        self.img_dir = img_dir
        self.transform = transform
        # self.target_transform = target_transform

    # the __len__ returns the number of samples in our dataset.
    def __len__(self):  # noting the number of labels
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,
                                self.img_labels.iloc[index, 0])
        # image = read_image(img_path)  # how is this reading image with label name in its path
        image = io.imread(img_path)
        label = torch.Tensor(int(self.img_labels.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label
