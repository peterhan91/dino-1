import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, path_to_images, csv_path='./csv/split_.csv', fold='train', sample=0, transform=None):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv(csv_path)
        self.fold = fold
        self.df = self.df[self.df['fold'] == fold]
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(frac=sample, random_state=42)
            print('subsample the training set with ratio %f' % sample)
        self.files = self.df['path'].tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.path_to_images, 
                        os.path.basename(self.files[idx])))
        if self.transform:
            out = self.transform(image)
        return out



class PatchLabelDataset(Dataset):
    def __init__(self, path_to_images, csv_path='./csv/diabetic.csv', fold='train', sample=0, transform=None):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv(csv_path)
        self.fold = fold
        self.df = self.df[self.df['fold'] == fold]
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(frac=sample, random_state=42)
            print('subsample the training set with ratio %f' % sample)
        self.files = self.df['path'].tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = os.path.basename(self.files[idx])
        image = Image.open(os.path.join(self.path_to_images, fname))
        llist = self.df[self.df['image']==fname]['score'].tolist()
        try:
            assert len(llist) == 1
        except AssertionError:
            print(fname, len(llist), llist)
        label = llist[0]
        if self.transform:
            out = self.transform(image)
        return out, label