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

# CT data has Gaussian distributed noise
class AddGaussianNoise(object):

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, data):
        noisy_image = data + np.random.normal(loc=0.0, scale=self.sigma, size=data.shape)

        return noisy_image


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


# Compose the image transforms for our dataset
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

    # AFFINE TRANSFORMATION
    affine_transform = tio.RandomAffine(scales=(0.7, 1.4),
                                        degrees=(-180, 180),
                                        isotropic=True,
                                        label_keys="label",
                                        include=["image", "label"],
                                        p=0.2)
    transformed_subject = affine_transform(subject)

    # GAUSSIAN NOISE
    noise = tio.RandomNoise(p=0.15, std=(0, 0.1), include=["image"])
    transformed_subject = noise(transformed_subject)

    # GAUSSIAN BLUR (width of kernel is in mm)
    blur = tio.RandomBlur(p=0.1, std=(1.25, 3.75), include=["image"])
    transformed_subject = blur(transformed_subject)

    # BRIGHTNESS MODIFICATION
    n = np.random.uniform()
    if n < 0.15:
        # Randomly select the brightness multiplier
        m = np.random.uniform(0.7, 1.3)
        min = torch.min(transformed_subject["image"]).detach().numpy()
        max = torch.max(transformed_subject["image"]).detach().numpy()

        scale = tio.RescaleIntensity(out_min_max=(m * min, m * max), include=["image"])
        transformed_subject = scale(transformed_subject)

    # CONTRAST MODIFICATION
    n = np.random.uniform()
    if n < 0.15:
        # Randomly select the brightness multiplier
        m = np.random.uniform(0.7, 1.3)
        min = torch.min(transformed_subject["image"])
        max = torch.max(transformed_subject["image"])

        scale = tio.RescaleIntensity(out_min_max=(m * min.detach().numpy(), m * max.detach().numpy()), include=["image"])
        transformed_subject = scale(transformed_subject)

        # Clip back to original value range
        clamp = tio.Clamp(out_min=min, out_max=max, include=["image"])
        transformed_subject = clamp(transformed_subject)

    # LOW-RES SIMULATION
    n = np.random.uniform()
    if n < 0.125:
        # first downsample, then upsample
        downsample = tio.Resize(target_shape=(256, 256, 1), include=["image", "label"])
        upsample = tio.Resize(target_shape=(512, 512, 1), include=["image", "label"])

        transformed_subject = downsample(transformed_subject)
        transformed_subject = upsample(transformed_subject)


    # GAMMA AUGMENTATION
    n = np.random.uniform()
    if n < 0.15:
        # need to scale intensities to range [0,1] first, and then back after transform
        # find initial min/max
        min = np.min(transformed_subject["image"].detach().numpy())
        max = np.max(transformed_subject["image"].detach().numpy())

        # set up operations
        scalePre = tio.RescaleIntensity(out_min_max=(0, 1), include=["image"])
        gamma = tio.RandomGamma(p=1, log_gamma=(0.7, 1.5), include=["image"])
        scalePost = tio.RescaleIntensity(out_min_max=(min, max), include=["image"])

        # Apply to sample
        transformed_subject = scalePre(transformed_subject)
        transformed_subject = gamma(transformed_subject)
        transformed_subject = scalePost(transformed_subject)

    # MIRRORING (RANDOM FLIP)
    flip = tio.RandomFlip(axes=(0, 1), flip_probability=0.5, include=["image", "label"])
    transformed_subject = flip(transformed_subject)

    transformed_img = transformed_subject["image"]
    transformed_lab = transformed_subject["label"]

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
        if True:
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