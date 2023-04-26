# A script to train a UNet using
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from datetime import datetime as dt

# local imports
from dataset import create_dataset
from loss import get_dice_per_class, dice_coeff
from UNet import UNet

# argparse
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--config_file", default="unet_v5_2_config.pkl", help="Configuration file for training")
parser.add_argument("-s", "--slurm", default=False, help="Running on SLURM?")
args = vars(parser.parse_args())

# set up global variables
CONFIG_FILE = args["config_file"]
SLURM = args['slurm']

# set flag to time operations
TIME = True

# Set up directories and filenames
if SLURM:
    root_dir = '/vol/biomedic3/kc2322/'
    code_dir = '/vol/biomedic3/kc2322/code/Experiment1/Experiment1'
else:
    root_dir = '/Users/katecevora/Documents/PhD'
    code_dir = '/Users/katecevora/Documents/PhD/code/Experiment1'

data_dir = os.path.join(root_dir, 'data/MSDPancreas2D/')
images_dir = os.path.join(data_dir, "imagesTr")
labels_dir = os.path.join(data_dir, "labelsTr")

save_path = os.path.join(root_dir, "models/MSDPancreas2D")

# Check if we have a GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train(train_loader, valid_loader, model_name, patch_size, batch_size, init_lr, num_epochs, num_classes):
    print("\n{}: Starting training.".format(dt.fromtimestamp(dt.now().timestamp())))
    start_time = dt.now()

    print("Saving model config at: {}".format(os.path.join(save_path, '{}_config.pkl'.format(model_name))))
    setup_dict = {"patch_size": patch_size,
                   "batch_size": batch_size}
    f = open(os.path.join(save_path, '{}_config.pkl'.format(model_name)), 'wb')
    pkl.dump(setup_dict, f)
    f.close()

    av_train_error = []
    av_train_dice = []
    av_valid_error = []
    av_valid_dice = []
    eps = []

    net = UNet(inChannels=1, outChannels=num_classes, imgSize=patch_size).to(device).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=init_lr, betas=(0.5, 0.999))
    optimizer.zero_grad()
    loss_BCE = nn.BCELoss()

    for epoch in range(num_epochs):
        ##########
        # Train
        ##########

        # set network to train prior to training loop
        net.train()  # this will ensure that parameters will be updated during training & that dropout will be used

        # reset the error log for the batch
        batch_train_error = []
        batch_train_dice = []

        # iterate over the batches in the training set
        for i, (data, label) in enumerate(train_loader):
            if i % 10 == 0:
                print("{}: Epoch {}, batch {}".format(dt.fromtimestamp(dt.now().timestamp()), epoch, i))
                print("{} Feeding data through network".format(dt.fromtimestamp(dt.now().timestamp())))

            optimizer.zero_grad()
            data = data.to(device).double()
            label = label.to(device).double()
            pred, _ = net(data)

            if i % 10 == 0:
                print("{} Calculating losses".format(dt.fromtimestamp(dt.now().timestamp())))

            # calculate loss
            L_dc = - dice_coeff(pred[:, 1:, :, :], label[:, 1:, :, :])
            L_ce = loss_BCE(pred, label)
            err = L_dc + L_ce

            if i % 10 == 0:
                print("{} Backpropagating losses".format(dt.fromtimestamp(dt.now().timestamp())))

            err.backward()
            optimizer.step()

            # append to the batch errors
            batch_train_error.append(err.item())
            batch_train_dice.append(-L_dc.item())

        print("{} Finished iterating over data. Saving model.".format(dt.fromtimestamp(dt.now().timestamp())))

        # Checkpoint model
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': err,
        }, os.path.join(save_path, '{}.pt'.format(model_name)))

        # #############
        # # Validation
        # #############
        batch_valid_error = []
        batch_valid_dice = []

        # set network to eval prior to training loop
        print("{}: Running evaluation.....".format(dt.fromtimestamp(dt.now().timestamp())))
        net.eval()
        for i, (data, label) in enumerate(valid_loader):
            data = data.to(device).double()
            label = label.to(device).double()
            pred, _ = net(data)

            # calculate validation loss
            dice_per_class = get_dice_per_class(pred, label)
            L_dc = - dice_coeff(pred[:, 1:, :, :], label[:, 1:, :, :])
            L_ce = loss_BCE(pred, label)
            err = L_dc + L_ce

            batch_valid_error.append(err.item())
            batch_valid_dice.append(-L_dc.item())

        print("{}: Finished evaluation.".format(dt.fromtimestamp(dt.now().timestamp())))

        # Calculate the average training and validation error for this epoch and store
        av_train_error.append(np.mean(np.array(batch_train_error)))
        av_train_dice.append(np.mean(np.array(batch_train_dice)))
        av_valid_error.append(np.mean(np.array(batch_valid_error)))
        av_valid_dice.append(np.mean(np.array(batch_valid_dice)))
        eps.append(epoch)

        print("{}: Saving losses.".format(dt.fromtimestamp(dt.now().timestamp())))

        # Save everything
        f = open(os.path.join(save_path, "{}_losses.pkl".format(model_name)), "wb")
        pkl.dump([eps, av_train_error, av_train_dice, av_valid_error, av_valid_dice], f)
        f.close()

        print("{}: Finished epoch".format(dt.fromtimestamp(dt.now().timestamp())))
        print('Epoch: {0}, train error: {1:.3f}, valid error: {2:.3f}'.format(eps[-1], av_train_error[-1],
                                                                              av_valid_error[-1]))
        time_elapsed = dt.now() - start_time
        time_elapsed_s = time_elapsed.total_seconds()
        time_elapsed_h = divmod(time_elapsed_s, 3600)[0]
        print("Time elasped since start of training: {}".format(time_elapsed_h))
        #print('Average dice for training batch:')
        #print('Average dice for validation batch:')


def evaluate(test_loader):
    # evaluate model performance on the test dataset
    return 1


def main():
    # Check if we have a GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # Try to load the configuration file specified
    try:
        f = open(os.path.join(code_dir, "config", CONFIG_FILE), 'rb')
        config_dict = pkl.load(f)
        f.close()
    except:
        print("Config file {} cannot be found".format(CONFIG_FILE))

    PATCH_SIZE = config_dict["patch_size"]
    if SLURM:
        BATCH_SIZE = config_dict['batch_size']
    else:
        BATCH_SIZE = 2
    NUM_WORKERS = config_dict["num_workers"]
    NUM_EPOCHS = config_dict['num_epochs']
    INIT_LEARNING_RATE = config_dict["init_lr"]
    TRAIN_PROP = config_dict["train_prop"]
    MODEL_NAME = config_dict["model_name"]
    FOLD = config_dict["fold"]
    AUGMENTATIONS = config_dict["augmentations"]
    NUM_CLASSES = 3

    print("Configuration:")
    print("Device: ", device)
    print("SLURM: {}".format(SLURM))
    print("Model name: {}".format(MODEL_NAME))
    print("Root dir: {}".format(root_dir))
    print("Path size: {}".format(PATCH_SIZE))
    print("Batch size: {}".format(BATCH_SIZE))
    print("Fold: {}".format(FOLD))
    print("Number of epochs: {}".format(NUM_EPOCHS))
    print("Augmentations: {}".format(AUGMENTATIONS))

    train_loader, valid_loader = create_dataset(root_dir, data_dir, FOLD, BATCH_SIZE, NUM_WORKERS, PATCH_SIZE, AUGMENTATIONS, NUM_CLASSES)

    # Train the network
    train(train_loader, valid_loader, MODEL_NAME, PATCH_SIZE, BATCH_SIZE, INIT_LEARNING_RATE, NUM_EPOCHS, NUM_CLASSES)



if __name__ == '__main__':
    main()

