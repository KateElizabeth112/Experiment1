import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


# Display a 2D image along one plane
def Display2D(image, vox_size, plane=0, save_path=""):
    # get centre voxel
    s = np.array(image.shape)
    c = np.around(s / 2).astype(int)

    if plane==0:
        image_slice = image[c[0], ::-1, :].transpose()
        aspect = vox_size[2] / vox_size[1]
    elif plane==1:
        image_slice = image[::-1, c[1], :].transpose()
        aspect = vox_size[2] / vox_size[1]
    elif plane==2:
        image_slice = image[::-1, ::-1, c[2]].transpose()
        aspect = vox_size[2] / vox_size[1]
    else:
        print("Please choose a plane within range")
        return -1

    plt.imshow(image_slice, cmap='gray', aspect=aspect)
    plt.title("Slices = {}".format(s[0]))
    plt.axis("off")

    if save_path=="":
        plt.show()
    else:
        plt.savefig(save_path)


# Display a 2D image along one plane, with segmentations overlaid on top
def DisplayOverlay2D(image, labels, vox_size, plane=0, save_path=""):
    # get centre voxel
    s = np.array(image.shape)
    c = np.around(s / 2).astype(int)
    max = 2    # This is the maximum value of the labels

    # Note that it is necessary to transpose the numpy array before plotting with imshow as imshow places the first
    # dimension on the Y-axis
    if plane==0:
        # Sagittal  (X-plane)
        image_slice = image[c[0], ::-1, :].transpose()
        labels_slice = labels[c[0], ::-1, :].transpose()
        aspect = vox_size[2] / vox_size[1]
    elif plane==1:
        # Coronal (Y-plane)
        image_slice = image[::-1, c[1], :].transpose()
        labels_slice = labels[::-1, c[1], :].transpose()
        aspect = vox_size[2] / vox_size[0]
    elif plane==2:
        # Transverse/Axial (Z-plane)
        image_slice = image[::-1, ::-1, c[2]].transpose()
        labels_slice = labels[::-1, ::-1, c[2]].transpose()
        aspect = vox_size[1] / vox_size[0]
    else:
        print("Please choose a plane within range")
        return -1

    alpha_array = np.zeros(labels_slice.shape)
    alpha_array[labels_slice > 0] = 0.5

    plt.imshow(image_slice, cmap='gray', aspect=aspect)
    plt.imshow(labels_slice, cmap='jet', alpha=alpha_array, vmin=0, vmax=max, aspect=aspect)
    plt.title("Num voxels = {}".format(s) + ", voxel size = ({0:.2f}, {1:.2f}, {2:.2f}) mm".format(vox_size[0],
                                                                                                   vox_size[1],
                                                                                                   vox_size[2]))
    plt.axis('off')


    if save_path=="":
        plt.show()
    else:
        plt.savefig(save_path)


# Display the central slice of a 3D scan along 3 axes
def Display3D(image):
    plt.figure(figsize=(10, 3))

    fontsize = 12

    # get centre voxel
    s = np.array(image.shape)
    c = np.around(s / 2).astype(int)

    plt.subplot(131)
    plt.imshow(image[c[0], ::-1, :], cmap='gray')
    #plt.axis('equal')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(image[::-1, c[1], :], cmap='gray', aspect=s[2]/s[0])
    #plt.axis('equal')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(image[::-1, ::-1, c[2]], cmap='gray', aspect=s[1]/s[0])
    #plt.axis('equal')
    plt.axis('off')

    plt.show()


# Display registration results (source, target, warped image) in two dimensions
def DisplayRegistration2D(source, target, warped, vox_spacing, save_path=""):
    plt.figure(figsize=(10, 5))

    fontsize = 12

    plt.suptitle("(Target voxel size: {0:.2f}, {1:.2f}, {2:.2f})".format(vox_spacing[0], vox_spacing[1], vox_spacing[2]))

    plt.subplot(131)
    plt.imshow(source[int(source.shape[0] / 2), ::-1, :], cmap='gray')
    plt.title('Source', fontsize=fontsize)
    plt.axis('equal')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(target[int(target.shape[0] / 2), ::-1, :], cmap='gray')
    plt.title('Target', fontsize=fontsize)
    plt.axis('equal')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(warped[int(warped.shape[0] / 2), ::-1, :], cmap='gray')
    plt.title('Warped', fontsize=fontsize)
    plt.axis('equal')
    plt.axis('off')

    if save_path == "":
        plt.show()
    else:
        plt.savefig(save_path)