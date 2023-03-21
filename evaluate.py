# scripts to evaluate segmentations
# evaulate model performance on the test set
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import torch
from loss import get_dice_per_class
from UNet import UNet
from dataset import create_test_dataset
from display import PlotSliceAndPrediction
#from monai.metrics import compute_surface_dice



ROOT_DIR = '/Users/katecevora/Documents/PhD'
DATA_DIR = os.path.join(ROOT_DIR, 'data/MSDPancreas2D/')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'images/test')
MODEL_DIR = os.path.join(ROOT_DIR, "models/MSDPancreas2D")
MODEL_NAME = "unet_v1_0.pt"
FOLD = "0"
NUM_CHANNELS = 2

organs_dict = {0: "background",
               1: "pancreas",
               2: "tumor"}

colors = ["#ffa07a", "#663d61", "#ed90e1", "#008b45", "#0f52ba", "#fa8072", "#15f4ee", "#4cbb17", "#fdff00", "#ff1493",
          "#9400d3", "#00ced1", "#d63a0f", "#3fff00"]


def get_surface_dice(y_pred, y, class_thresholds):
    # covert predictions to one hot encoding
    max_idx = torch.argmax(y_pred, 1, keepdim=True)
    one_hot = torch.FloatTensor(y_pred.shape)
    one_hot.zero_()
    one_hot.scatter_(1, max_idx, 1)
    res = compute_surface_dice(one_hot, y, class_thresholds, include_background=True, distance_metric='euclidean')

    res = res.numpy()
    res = res.reshape(-1)
    return res


def evaluate(test_loader, model_path, fold, ds_length):
    # Check if we have a GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load the model
    net = UNet(inChannels=1, outChannels=NUM_CHANNELS).to(device).double()

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    net.eval()

    # load the filenames of test data
    #f = open(os.path.join(root_dir, "filenames_ts.pkl"), 'rb')
    #filenames_ts = pkl.load(f)
    #f.close()

    # create an empty array to store results
    dice_all = np.zeros((NUM_CHANNELS, ds_length))
    nsd_all = np.zeros((NUM_CHANNELS, ds_length))

    # Make a new folder to store the output images
    try:
        os.mkdir(os.path.join(OUTPUT_DIR, MODEL_NAME.split(".")[0]))
    except:
        print("Output directory already exists")

    for j, (data, lab) in enumerate(test_loader):
        pred = net(data.to(device))  # shape (B, C, H, W)

        # convert preds to one-hot so it's comparable with output from nnU-Net
        max_idx = torch.argmax(pred, 1, keepdim=True)
        one_hot = torch.FloatTensor(pred.shape)
        one_hot.zero_()
        one_hot.scatter_(1, max_idx, 1)

        dice = get_dice_per_class(one_hot, lab.to(device)).cpu().detach().numpy()
        #nsd = get_surface_dice(one_hot, lab.to(device), [1.5 for i in range(14)])

        # Fill Dice and NSD array
        dice_all[:, j] = dice

        pred = one_hot.cpu().detach().numpy()
        lab = lab.cpu().detach().numpy()
        img = data.cpu().detach().numpy()

        # lose the first two image dimensions for plotting
        pred = np.squeeze(pred)[1, :, :]
        lab = np.squeeze(lab)[1, :, :]
        img = np.squeeze(img)

        if False:
            # Visualise
            PlotSliceAndPrediction(img, lab, pred, save_path=os.path.join(OUTPUT_DIR,
                                                                          MODEL_NAME.split(".")[0],
                                                                          "{}.png".format(j)))

    # Save results
    f = open(os.path.join(OUTPUT_DIR, MODEL_NAME.split(".")[0], "results.pkl"), 'wb')
    pkl.dump([dice_all, nsd_all], f)
    f.close()

    print('Done')






def main():
    test_loader, ds_length = create_test_dataset(DATA_DIR)
    evaluate(test_loader, os.path.join(MODEL_DIR, MODEL_NAME), FOLD, ds_length)


if __name__ == "__main__":
    main()