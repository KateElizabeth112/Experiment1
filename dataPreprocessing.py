# Functions for data pre-processing like normalisation
import os
import random
import numpy as np
import nibabel as nib
#import SimpleITK as sitk
import matplotlib.pyplot as plt
import copy


root_dir = '/Users/katecevora/Documents/PhD'
CLIP = False

def getGlobalMeanAndSTD(path_images, path_labels,  n_samples):
    """
    Estimate the global mean and standard deviation of foreground pixels in a dataset through random sampling.
    :param path:
    :return:
    """
    filenames = os.listdir(path_images)

    # randomly sample some filenames
    samples = random.sample(filenames, n_samples)

    stack = []

    for f in samples:
        if f.endswith(".nii.gz"):
            img_raw = nib.load(os.path.join(path_images, f)).get_fdata()

            # Extract the name so we can open the label file
            name = f.split(".")[0][:-5]
            f_label = name + ".nii.gz"

            label_raw = nib.load(os.path.join(path_labels, f_label)).get_fdata()

            # Can we create a decent background mask?
            label_raw[label_raw>1] = 1

            #plt.clf()
            #plt.imshow(img_raw, cmap='gray')
            #plt.imshow(mask, alpha=0.6, cmap='jet')
            #plt.show()

            # Multiply image with background mask to get foreground pixels
            img_raw_masked = img_raw * label_raw

            # Flatten the pixel values
            img_raw_flat = img_raw_masked.flatten()

            # Stack
            stack.extend(img_raw[label_raw == 1].tolist())


    # Find mean and standard deviation
    mu = np.mean(np.array(stack))
    sigma = np.std(np.array(stack))

    # get percentiles
    percentiles = np.percentile(np.array(stack), [2.5, 99.5])

    return mu, sigma, percentiles



def clipAndNormalise(input_path, output_path, mu, sigma, percentiles):
    """
    For preparing CT data. First images are clipped between the 50 and 99.5 percentiles of foreground voxel values.
    Then, images are z-normed using global mean and standard deviation values
    :return:
    """
    filenames = os.listdir(input_path)

    for f in filenames:
        if f.endswith(".nii.gz"):
            img = nib.load(os.path.join(input_path, f))
            img_raw = img.get_fdata()

            # First clip
            if CLIP:
                img_raw_clipped = copy.deepcopy(img_raw)
                img_raw_clipped[img_raw < percentiles[0]] = percentiles[0]
                img_raw_clipped[img_raw > percentiles[1]] = percentiles[1]
            else:
                img_raw_clipped = copy.deepcopy(img_raw)


            # Then normalise
            img_raw_normed = (img_raw_clipped - mu) / sigma

            if True:
                plt.clf()
                plt.subplot(1, 3, 1)
                plt.imshow(img_raw_clipped, cmap='gray')
                plt.subplot(1, 3, 2)
                plt.imshow(img_raw_normed, cmap='gray')
                plt.subplot(1, 3, 3)
                plt.imshow(img_raw, cmap='gray')
                plt.show()

                plt.clf()
                plt.subplot(1, 3, 1)
                plt.hist(img_raw_clipped.flatten())
                plt.subplot(1, 3, 2)
                plt.hist(img_raw_normed.flatten())
                plt.subplot(1, 3, 3)
                plt.hist(img_raw.flatten())
                plt.show()

            # And save in the new location
            #img_sitk = sitk.GetImageFromArray(img_raw_normed)
            #sitk.WriteImage(img_sitk, os.path.join(output_path, f))

            clipped_img = nib.Nifti1Image(img_raw_normed, img.affine, img.header)
            nib.save(clipped_img, os.path.join(output_path, f))

def main():

    mu, sigma, percentiles = getGlobalMeanAndSTD(os.path.join(root_dir, "data/MSDPancreas2D/imagesTr/"),
                                                 os.path.join(root_dir, "data/MSDPancreas2D/labelsTr/"),
                                                 150)

    clipAndNormalise(os.path.join(root_dir, "data/MSDPancreas2D/imagesTs/"),
                     os.path.join(root_dir, "data/MSDPancreas2D/preprocessed/imagesTs/"),
                     mu,
                     sigma,
                     percentiles)


if __name__ == "__main__":
    main()