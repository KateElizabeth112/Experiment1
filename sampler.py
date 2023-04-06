# Functions to enable patch sampling
import numpy as np


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
