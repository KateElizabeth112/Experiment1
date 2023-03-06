# Generate the indices for 5-fold cross validation. For each fold, the indicies are retrieved and used to create the
# train/validation split, which is passed to the dataloaders
import pickle as pkl
import os
import numpy as np

NUM_FOLDS = 5
TRAINING_DIR = '/Users/katecevora/Documents/PhD/data/MSDPancreas2D/imagesTr'
OUTPUT_DIR = '/Users/katecevora/Documents/PhD/params/MSDPancreas2D'


def main():
    # find filenames of training dataset (make sure we only include .nii.gz files)
    filenames = os.listdir(TRAINING_DIR)
    for f in filenames:
        if not(f.endswith(".nii.gz")):
            filenames.remove(f)

    # get training dataset length
    ds_len = len(filenames)
    fold_len = int(np.floor(ds_len / NUM_FOLDS))
    train_len = (NUM_FOLDS - 1) * fold_len

    print("Generating indicies for {} dataset folds...".format(NUM_FOLDS))
    print("Length of dataset: {}".format(ds_len))
    print("Number of training images per fold: {}".format(train_len))

    # randomly assign indices to folds
    indices = np.arange(0, ds_len)
    folds = []
    for f in range(NUM_FOLDS):
        idx = np.random.choice(range(indices.shape[0]), fold_len, replace=False)
        folds.append(indices[idx].astype('int'))
        indices = np.delete(indices, idx)

    # combine folds into training and validation sets
    train_indices = []
    valid_indices = []
    for f in range(NUM_FOLDS):
        valid_indices.append(folds[f])
        train_idx = np.zeros(train_len)
        current_idx = 0
        for k in range(NUM_FOLDS):
            if (f != k):
                train_idx[current_idx:current_idx+fold_len] = folds[k]
                current_idx += fold_len
        train_indices.append(train_idx)

    # save the folds
    f = open(os.path.join(OUTPUT_DIR, "fold.pkl"), "wb")
    pkl.dump([train_indices, valid_indices], f)
    f.close()




if __name__ == "__main__":
    main()