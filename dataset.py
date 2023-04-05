# File to contain dataset class
import copy
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

SAMPLING_STRATEGY = "weighted"

# CT data has Gaussian distributed noise
class AddGaussianNoise(object):

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, data):
        noisy_image = data + np.random.normal(loc=0.0, scale=self.sigma, size=data.shape)

        return noisy_image


def samplePatchWeighted(image, label, probs, patch_size):
    """
    Function to sample a patch from an image with specified probabilities
    :param image: the raw image.
    :param label: the label map for the image.
    :param probs: the probability with which we will sample this class as a list. Length must match number of classes.
    :param patch_size: The size of the patch which will be sampled.
    :return:
    """

    # Check that the probabilities list matches the number of classes.
    if not(np.unique(label).shape[0] == len(probs)):
        raise(Exception("The length of the probabilites list does not match the number of label classes"))

    if not(np.sum(np.array(probs)) == 1):
        raise(Exception("The probabilities provided do not sum to 1."))

    # Choose which class will be at the centre of the patch using the probabilities list
    c = np.argmax(np.random.multinomial(1, probs))

    # Choose the relevant part of the label map, convert to verticies and crop so that patch is within image limits
    verts = np.nonzero(label[c, :, :] == c)

    cx = np.unique(verts[:, 0])
    cy = np.unique(verts[:, 1])

    delta = int(np.floor(patch_size / 2))
    vert_min = delta
    vert_max = image.shape[1] - delta

    cx_cropped = cx[(cx > vert_min) & (cx < vert_max)]
    cy_cropped = cy[(cy > vert_min) & (cy < vert_max)]

    # Check if this has resulted in no suitable vertices
    if (cy_cropped.shape[0] < 1) or (cx_cropped.shape[0] < 1):
        y = 256
        x = 256
    else:
        # Randomly sample to get the central verticies of the patch
        x = int(np.random.choice(cx_cropped, size=1))
        y = int(np.random.choice(cy_cropped, size=1))

    # Crop the patch from the image and label
    image_cropped = image[:, x-delta:x+delta, y-delta:y+delta]
    label_cropped = label[:, x-delta:x+delta, y-delta:y+delta]

    if np.min(cx_cropped) - delta < 0:
        print(cx_cropped)
        raise(Exception("x coordinate is less than minimum"))
    if np.min(cy_cropped) - delta < 0:
        print(cy_cropped)
        raise(Exception("y coordinate is less than minimum"))
    if np.max(cx_cropped) + delta > 512:
        print(cx_cropped)
        raise (Exception("x coordinate is greater than maximum"))
    if np.max(cy_cropped) + delta > 512:
        print(cy_cropped)
        raise (Exception("y coordinate is greater than maximum"))

    if image_cropped.shape[2] < 256:
        print(image_cropped.shape)
        print(x), print(y)
        print(cy_cropped), print(cx_cropped)

    return image_cropped, label_cropped


def samplePatchRandom(image, label, patch_size):
    """
    Randomly sample a patch from an image and return cropped label and patch/
    :param image:
    :param label:
    :return:
    """
    img_size = image.shape

    maxW = img_size[2] - patch_size
    maxH = img_size[1] - patch_size

    # randomly select patch origin
    xO = np.random.randint(0, maxH)
    yO = np.random.randint(0, maxW)

    # Select patch
    image_patch = image[:, xO:xO + patch_size, yO:yO + patch_size]
    label_patch = label[:, xO:xO + patch_size, yO:yO + patch_size]

    return image_patch, label_patch


def clipAndNormalise(image, mu, sigma, percentiles):
    """
    For preparing CT data. First images are clipped between the 50 and 99.5 percentiles of foreground voxel values.
    Then, images are z-normed using global mean and standard deviation values
    :return:
    """
    # First clip
    CLIP = True
    if CLIP:
        img_raw_clipped = copy.deepcopy(image)
        img_raw_clipped[image < percentiles[0]] = percentiles[0]
        img_raw_clipped[image > percentiles[1]] = percentiles[1]
    else:
        img_raw_clipped = copy.deepcopy(image)

    # Then normalise
    img_raw_normed = (img_raw_clipped - mu) / sigma

    if False:
        plt.clf()
        plt.subplot(1, 3, 2)
        plt.imshow(img_raw_clipped[:, :, 0], cmap='gray')
        plt.title("Clipped")
        plt.subplot(1, 3, 3)
        plt.imshow(img_raw_normed[:, :, 0], cmap='gray')
        plt.title("Normed")
        plt.subplot(1, 3, 1)
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.title("Original")
        plt.show()

        plt.clf()
        plt.subplot(1, 3, 2)
        plt.hist(img_raw_clipped.flatten())
        plt.title("Clipped")
        plt.subplot(1, 3, 3)
        plt.hist(img_raw_normed.flatten())
        plt.title("Normed")
        plt.subplot(1, 3, 1)
        plt.hist(image.flatten())
        plt.title("Original")
        plt.show()

    return img_raw_normed


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

        # Expand the label to the number of channels so we can use one-hot encoding
        lab_full = np.zeros((lab.shape[0], lab.shape[1], self.num_channels))

        for c in range(self.num_channels):
            lab_full[:, :, c][lab[:, :, 0] == c] = 1

        # swap channels to the first dimension as pytorch expects
        # shape (C, H, W)
        img = torch.tensor(np.swapaxes(img, 0, 2)).double()
        lab_full = torch.tensor(np.swapaxes(lab_full, 0, 2)).double()

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
            plt.show()

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
