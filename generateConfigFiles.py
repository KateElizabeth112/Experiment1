# Script to generate configuration files for experiment runs
import pickle as pkl
import os

MODEL_NAME = "unet_v6_10"
PATCH_SIZE = 256
BATCH_SIZE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 150
INIT_LEARNING_RATE = 3e-4
TRAIN_PROP = 0.8
FOLD = 0
# ['affine', 'gaussian_noise', "gaussian_blur", "brightness", "contrast", "low-res", "gamma", "flip"]
AUGMENTATIONS = ['affine', 'gaussian_noise', "gaussian_blur", "brightness", "contrast", "low-res", "gamma", "flip"]


def main():
    # Create the config dictionary
    config_dict = {"model_name": MODEL_NAME,
                   "patch_size": PATCH_SIZE,
                   "batch_size": BATCH_SIZE,
                   "num_workers": NUM_WORKERS,
                   "num_epochs": NUM_EPOCHS,
                   "init_lr": INIT_LEARNING_RATE,
                   "train_prop": TRAIN_PROP,
                   "fold": FOLD,
                   "augmentations": AUGMENTATIONS}

    # Save as a pickle
    f = open(os.path.join('./config', "{}_config.pkl".format(MODEL_NAME)), "wb")
    pkl.dump(config_dict, f)
    f.close()


if __name__ == "__main__":
    main()