# File to contain dataset class
import os
import nibabel as nib
import numpy as np
import pickle as pkl
import torch
import SimpleITK as sitk

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset


# CT data has Gaussian distributed noise
class AddGaussianNoise(object):

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, data):
        noisy_image = data + np.random.normal(loc=0.0, scale=self.sigma, size=data.shape)

        return noisy_image


def create_elastic_deformation(image_shape, num_controlpoints, sigma):
    # Create an instance of a SimpleITK image of the same size as our image
    itkimg = sitk.GetImageFromArray(np.zeros(image_shape))

    # This parameter is just a list with the number of control points per image dimensions
    trans_from_domain_mesh_size = [num_controlpoints] * itkimg.GetDimension()

    # We initialise the transform here: Passing the image size and the control point specifications
    bspline_transformation = sitk.BSplineTransformInitializer(itkimg, trans_from_domain_mesh_size)

    # Isolate the transform parameters: the bspline control points and spline coefficients
    params = np.asarray(bspline_transformation.GetParameters(), dtype=float)

    # Let's initialise the transform by randomly displacing each control point by a random distance (magnitude sigma)
    params = np.random.randn(params.shape[0]) * sigma
    bspline_transformation.SetParameters(tuple(params))

    return bspline_transformation


class ApplyElasticDeformation(object):

    def __init__(self, sigma, num_control_points):
        # strength of transformation
        self.sigma = sigma

        # granularity of transformation
        self.num_control_points = num_control_points

    def __call__(self, image, label):
        # We need to choose an interpolation method for our transformed image, let's just go with b-spline
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkBSpline)

        # create empty container for output image
        warped_img = np.zeros(image.shape)

        # transform each channel with the same deformation
        # we are trying to simulate variations in brain structure here
        n_channels = image.shape[0]
        image_shape = image[0, :, :].shape

        # Initialise the transform
        bspline_transform = create_elastic_deformation(image_shape, self.num_control_points, self.sigma)

        for i in range(n_channels):
            # Let's convert our image to an sitk image
            single_channel_img = image[i, :, :]
            sitk_image = sitk.GetImageFromArray(single_channel_img)
            # sitk_grid = create_grid(image)

            # Specify the image to be transformed: This is the reference image
            resampler.SetReferenceImage(sitk_image)
            resampler.SetDefaultPixelValue(0)

            # Set the transform in the initialiser
            resampler.SetTransform(bspline_transform)

            # Carry out the resampling according to the transform and the resampling method
            out_img_sitk = resampler.Execute(sitk_image)
            # out_grid_sitk = resampler.Execute(sitk_grid)

            # Convert the image back into a python array
            out_img = sitk.GetArrayFromImage(out_img_sitk)

            # store channel in output container
            warped_img[i, :, :] = out_img.reshape(single_channel_img.shape)

        return warped_img


# Compose the image transform for our dataset
def ImageTranform(img, label):
    # Add Gaussian noise of  magnitude sigma
    add_noise = AddGaussianNoise(sigma=0.1)
    noisy_img = add_noise(img)

    # Elastic deformation of image
    return noisy_img, label


class MSDPancreas(Dataset):
    def __init__(self, root_dir, num_channels, patch_size, train=True, transform=None):
        '''
          root_dir - string - path towards the folder containg the data
        '''
        # Save the root_dir as a class variable
        self.root_dir = root_dir
        if train:
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

        # Expand the label to the number of channels so we can use one-hot encoding
        lab_full = np.zeros((lab.shape[0], lab.shape[1], self.num_channels))

        for c in range(self.num_channels):
            lab_full[:, :, c][lab[:, :, 0] == c] = 1

        # swap channels to the first dimension as pytorch expects
        # shape (C, H, W)
        #img = torch.tensor(np.swapaxes(img, 0, 2)).double()
        img = torch.tensor(img).double()
        lab_full = torch.tensor(np.swapaxes(lab_full, 0, 2)).double()

        # Randomly crop if we are using patch size < 512
        img_size = img.shape
        if self.patch_size < img_size[2]:
            maxW = img_size[2] - self.patch_size
            maxH = img_size[1] - self.patch_size

            # randomly select patch origin
            xO = np.random.randint(0, maxH)
            yO = np.random.randint(0, maxW)

            # Select patch
            img = img[:, xO:xO+self.patch_size, yO:yO+self.patch_size]
            lab_full = lab_full[:, xO:xO+self.patch_size, yO:yO+self.patch_size]

        # carry out dataset augmentations if the flag has been set
        if self.transform:
            img, lab_full = self.transform(img, lab_full)

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
    test_dataset = MSDPancreas(root_dir=data_dir, num_channels=2, train=False)

    ds_length = test_dataset.__len__()

    # Create a data loader
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1
    )

    return test_loader, ds_length
