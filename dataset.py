# File to contain dataset class
import os
import nibabel as nib
import numpy as np
import pickle as pkl
import torch
import torchio as tio
import SimpleITK as sitk
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset

from sampler import samplePatchRandom, samplePatchWeighted
from transforms import clipAndNormalise, augmentImage

SAMPLING_STRATEGY = "random"


class MSDPancreas(Dataset):
    def __init__(self, root_dir, num_channels, patch_size, train=True, transform=None):
        '''
          root_dir - string - path towards the folder containg the data
        '''
        # Save the root_dir as a class variable
        self.root_dir = root_dir
        self.train = train
        if self.train:
            self.train_imgs = os.path.join(root_dir, "imagesTr")
            self.train_labels = os.path.join(root_dir, "labelsTr")
            self.filenames = os.listdir(os.path.join(root_dir, "imagesTr"))
        else:
            self.train_imgs = os.path.join(root_dir, "imagesTs")
            self.train_labels = os.path.join(root_dir, "labelsTs")
            self.filenames = os.listdir(os.path.join(root_dir, "imagesTs"))

        self.num_channels = num_channels
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Fetch file filename
        img_name = self.filenames[idx]

        # Remove the '0000' required at the end of nnUNet train images to get the corresponding label name
        label_name = img_name.split('.')[0][:-5] + ".nii.gz"

        # Load the nifty image
        img = nib.load(os.path.join(self.train_imgs, img_name))
        lab = nib.load(os.path.join(self.train_labels, label_name))

        # Get the voxel values as a numpy array
        img = np.array(img.get_fdata())
        lab = np.array(lab.get_fdata())

        # Clip and normalise image according to global mean and standard deviation
        img = clipAndNormalise(img, 75.58018995449481, 68.39769344841305, [-61, 208])

        # Create tensors
        img = torch.tensor(img).double()
        lab = torch.tensor(lab).double()

        # Apply transforms
        img, lab = augmentImage(img, lab)

        # Expand the label to the number of channels so we can use one-hot encoding
        lab_full = np.zeros((lab.shape[0], lab.shape[1], self.num_channels))

        for c in range(self.num_channels):
            lab_full[:, :, c][lab[:, :, 0] == c] = 1

        # swap channels to the first dimension as pytorch expects
        # shape (C, H, W)
        img = torch.swapaxes(img, 0, 2)
        lab_full = torch.swapaxes(torch.tensor(lab_full).double(), 0, 2)

        # Select sampling strategy
        if self.patch_size < img.shape[1]:
            if SAMPLING_STRATEGY == "weighted":
                img, lab_full = samplePatchWeighted(img, lab_full, [0.3, 0.7], self.patch_size)
            elif SAMPLING_STRATEGY == "random":
                img, lab_full = samplePatchRandom(img, lab_full, self.patch_size)

        # Toggle to display sampled image
        if False:
            plt.clf()
            plt.imshow(img[0, :, :], cmap='gray')
            plt.imshow(lab_full[1, :, :], cmap='jet', alpha=0.4)
            plt.show()

        return img, lab_full


def create_dataset(root_dir, data_dir, fold, batch_size, num_workers, patch_size):
    # Create train and test datasets

    # load folds
    f = open(os.path.join(root_dir, "params/MSDPancreas2D/fold.pkl"), "rb")
    [train_indices, valid_indices] = pkl.load(f)
    f.close()

    # create a dataset
    dataset = MSDPancreas(root_dir=data_dir, num_channels=2, patch_size=patch_size, train=True)

    # create train and test datasets
    train_dataset = Subset(dataset, train_indices[fold].astype('int'))
    valid_dataset = Subset(dataset, valid_indices[fold].astype('int'))

    print("Number of training samples: {}".format(train_dataset.__len__()))
    print("Number of validation samples: {}".format(valid_dataset.__len__()))

    # Create the required DataLoaders for training and testing
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )

    return train_loader, valid_loader


def create_test_dataset(data_dir):
    # create just a test dataset
    test_dataset = MSDPancreas(root_dir=data_dir, num_channels=2, train=False, patch_size=512)

    ds_length = test_dataset.__len__()

    # Create a data loader
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1
    )

    return test_loader, ds_length
