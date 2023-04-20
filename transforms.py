# Functions for transformation and augmentation of images
import os
import copy
import numpy as np
import pickle as pkl
import torch
import torchio as tio
import SimpleITK as sitk
import matplotlib.pyplot as plt

import nibabel as nib


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


def randomAffine(subject, probability=0.2):
    affine_transform = tio.RandomAffine(scales=(0.7, 1.4),
                                        degrees=(-180, 180),
                                        isotropic=True,
                                        label_keys="label",
                                        include=["image", "label"],
                                        p=probability)
    subject = affine_transform(subject)

    return subject


def randomGaussianNoise(subject, probability=0.15):
    noise = tio.RandomNoise(p=probability, std=(0, 0.1), include=["image"])
    subject = noise(subject)

    return subject


def randomGaussianBlur(subject, probability=0.1):
    # width of kernel is in mm
    blur = tio.RandomBlur(p=0.1, std=(1.25, 3.75), include=["image"])
    subject = blur(subject)

    return subject


def randomBrightnessMul(subject, probability=0.15):
    n = np.random.uniform()
    if n < probability:
        # Randomly select the brightness multiplier
        m = np.random.uniform(0.7, 1.3)
        min = torch.min(subject["image"]).detach().numpy()
        max = torch.max(subject["image"]).detach().numpy()

        scale = tio.RescaleIntensity(out_min_max=(m * min, m * max), include=["image"])
        subject = scale(subject)

    return subject


def randomContrastMod(subject, probability=0.15):
    n = np.random.uniform()
    if n < probability:
        # Randomly select the brightness multiplier
        m = np.random.uniform(0.7, 1.3)
        min = torch.min(subject["image"])
        max = torch.max(subject["image"])

        scale = tio.RescaleIntensity(out_min_max=(m * min.detach().numpy(), m * max.detach().numpy()),
                                     include=["image"])
        subject = scale(subject)

        # Clip back to original value range
        clamp = tio.Clamp(out_min=min, out_max=max, include=["image"])
        subject = clamp(subject)

    return subject


def lowResSim(subject, probability=0.125):
    n = np.random.uniform()
    if n < probability:
        # first downsample, then upsample
        downsample = tio.Resize(target_shape=(256, 256, 1), include=["image", "label"])
        upsample = tio.Resize(target_shape=(512, 512, 1), include=["image", "label"])

        subject = downsample(subject)
        subject = upsample(subject)

    return subject


def randomGamma(subject, probability=0.15):
    n = np.random.uniform()
    if n < probability:
        # need to scale intensities to range [0,1] first, and then back after transform
        # find initial min/max
        min = np.min(subject["image"].detach().numpy())
        max = np.max(subject["image"].detach().numpy())

        # set up operations
        scalePre = tio.RescaleIntensity(out_min_max=(0, 1), include=["image"])
        gamma = tio.RandomGamma(p=1, log_gamma=(0.7, 1.5), include=["image"])
        scalePost = tio.RescaleIntensity(out_min_max=(min, max), include=["image"])

        # Apply to sample
        subject = scalePre(subject)
        subject = gamma(subject)
        subject = scalePost(subject)

    return subject


def randomFlip(subject, probability=0.5):
    flip = tio.RandomFlip(axes=(0, 1), flip_probability=probability, include=["image", "label"])
    subject = flip(subject)

    return subject


def augmentImage(image, label):
    """
    Image and label should have 4 dimensions each (C, H, W, D)
    :param image:
    :param label:
    :return:
    """

    # Expand the image and label to have four dimensions
    image = image.unsqueeze(0)
    label = label.unsqueeze(0)

    # Create a dictionary to feed to transform
    subject = {"image": image, "label": label}

    #augmentation_list = ['affine', 'gaussian_noise', "gaussian_blur", "brightness", "contrast", "low-res", "gamma", "flip"]
    augmentation_list = ["affine"]

    # AFFINE TRANSFORMATION
    if 'affine' in augmentation_list:
        subject = randomAffine(subject, probability=0.2)

    # GAUSSIAN NOISE
    if 'gaussian_noise' in augmentation_list:
        subject = randomGaussianNoise(subject, probability=0.15)

    # GAUSSIAN BLUR
    if "gaussian_blur" in augmentation_list:
        subject = randomGaussianBlur(subject, probability=0.1)

    # BRIGHTNESS MODIFICATION
    if "brightness" in augmentation_list:
        subject = randomBrightnessMul(subject, probability=0.15)

    # CONTRAST MODIFICATION
    if "contrast" in augmentation_list:
        subject = randomContrastMod(subject, probability=0.15)

    # LOW-RES SIMULATION
    if "low_res" in augmentation_list:
        subject = lowResSim(subject, probability=0.125)

    # GAMMA AUGMENTATION
    if "gamma" in augmentation_list:
        subject = randomGamma(subject, probability=0.15)

    # MIRRORING (RANDOM FLIP)
    if "flip" in augmentation_list:
        subject = randomFlip(subject, probability=0.5)

    transformed_img = subject["image"]
    transformed_lab = subject["label"]

    # squeeze before returning
    transformed_img = transformed_img.squeeze(0)
    transformed_lab = transformed_lab.squeeze(0)

    return transformed_img, transformed_lab


def main():

    # load an image from the train set
    root_dir = '/Users/katecevora/Documents/PhD'
    data_dir = os.path.join(root_dir, 'data/MSDPancreas2D/')
    img_path = os.path.join(data_dir, "imagesTr/pancreas_33135_0000.nii.gz")
    lab_path = os.path.join(data_dir, "labelsTr/pancreas_33135.nii.gz")

    img = nib.load(img_path)
    lab = nib.load(lab_path)

    # Get the voxel values as a numpy array
    img = np.array(img.get_fdata())
    img_tensor = torch.tensor(img).double()

    lab = np.array(lab.get_fdata())
    lab_tensor = torch.tensor(lab).double()

    for i in range(1000):
        transformed_img, transformed_lab = augmentImage(img_tensor, lab_tensor)

        # plot both
        if False:
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.imshow(img_tensor[:, :, 0], cmap='gray')
            plt.subplot(2, 2, 2)
            plt.imshow(transformed_img[:, :, 0], cmap='gray')
            plt.subplot(2, 2, 3)
            plt.imshow(lab_tensor[:, :, 0], cmap="jet")
            plt.subplot(2, 2, 4)
            plt.imshow(transformed_lab[:, :, 0], cmap="jet")
            plt.show()


if __name__ == "__main__":
    main()