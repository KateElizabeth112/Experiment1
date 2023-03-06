# Create the train/test split by randomly selecting patients
# We do not need to worry about
import numpy as np
import pickle as pkl
import os

root_dir = '/Users/katecevora/Documents/PhD/'
os.chdir(root_dir)

train_folder_base = 'data/MSDPancreas/imagesTr'
train_folder_2d = 'data/MSDPancreas2D/imagesTr'
test_folder_2d = 'data/MSDPancreas2D/imagesTs'
train_folder_labels_2d = 'data/MSDPancreas2D/labelsTr'
test_folder_labels_2d = 'data/MSDPancreas2D/labelsTs'
params_folder = 'params'

# Proportion of the dataset used for testing
TEST_PROP = 0.05

# set the random seed so get repeatable results
np.random.seed(0)


def createTrainTestSplit():
    # Create lists of candidates for the training and tests sets
    filenames = os.listdir(train_folder_base)
    candidates = []

    # Make sure we only have nib files in the list
    for f in filenames:
        if f.endswith('.nii.gz'):
            candidates.append(f.split('.')[0])

    num_candidates = len(candidates)
    num_test_samples = int(np.ceil(num_candidates * TEST_PROP))
    num_train_samples = num_candidates - num_test_samples
    print("Number of candidates: {0}".format(num_candidates))
    print("Number of test samples: {0}".format(num_test_samples))

    # Randomly create the split
    indicies = np.arange(0, num_candidates)
    np.random.shuffle(indicies)
    test_indicies = indicies[:num_test_samples]
    train_indicies = indicies[num_test_samples:]

    # Check arrays are the size we expect
    if test_indicies.shape[0] != num_test_samples:
        print("Number of test samples not as expected")
        return -1

    if train_indicies.shape[0] != num_train_samples:
        print("Number of train samples not as expected")
        return -1

    # Create lists of names
    test_candidates = np.array(candidates)[test_indicies]
    train_candidates = np.array(candidates)[train_indicies]

    # Now save the lists
    f = open(os.path.join(params_folder, "MSDPancreas_train_test_names.pkl"), 'wb')
    pkl.dump([train_candidates, test_candidates], f)
    f.close()


def sort2DSlices():
    # Sort the 2D slices according to candidates in the train and test groups
    f = open(os.path.join(params_folder, "MSDPancreas_train_test_names.pkl"), 'rb')
    [train_candidates, test_candidates] = pkl.load(f)
    f.close()

    count = 0
    filenames = os.listdir(train_folder_2d)
    for f in filenames:
        name = f.split(".")[0][:12]
        label_f = f.split(".")[0][:-5] + ".nii.gz"
        if name in test_candidates:
            # Move
            os.rename(os.path.join(train_folder_2d, f), os.path.join(test_folder_2d, f))
            os.rename(os.path.join(train_folder_labels_2d, label_f), os.path.join(test_folder_labels_2d, label_f))
            count += 1

    print(count)


def main():
    sort2DSlices()
    print("Done")


if __name__ == "__main__":
    main()