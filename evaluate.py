# scripts to evaluate segmentations
# evaulate model performance on the test set
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import torch
import torchio as tio
from loss import get_dice_per_class
from UNet import UNet
from dataset import create_test_dataset
from display import PlotSliceAndPrediction
#from monai.metrics import compute_surface_dice


ROOT_DIR = '/Users/katecevora/Documents/PhD'
DATA_DIR = os.path.join(ROOT_DIR, 'data/MSDPancreas2D/')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'images/test')
MODEL_DIR = os.path.join(ROOT_DIR, "models/MSDPancreas2D")
MODEL_NAME = "unet_v5_3.pt"
FOLD = "0"
NUM_CHANNELS = 2
PATCH_OVERLAP = 128

organs_dict = {0: "background",
               1: "pancreas",
               2: "tumor"}

colors = ["#ffa07a", "#663d61", "#ed90e1", "#008b45", "#0f52ba", "#fa8072", "#15f4ee", "#4cbb17", "#fdff00", "#ff1493",
          "#9400d3", "#00ced1", "#d63a0f", "#3fff00"]


def visualiseLatent(latent_tensor, label, dice, save_path=""):
    img_size = 512

    # get rid of the second dimension
    latent_tensor = latent_tensor.squeeze()

    # flatten the last two dimensions
    latent_tensor = latent_tensor.flatten(start_dim=1).swapaxes(0, 1)
    label = label.flatten()

    # OPTIONAL: add spatial map to features
    x_map = (np.reshape(np.tile(np.arange(0, img_size), img_size), (img_size, img_size)) / 256) - 1
    y_map = np.swapaxes(x_map, 0, 1) / img_size

    x_map = np.expand_dims(x_map, 0)
    y_map = np.expand_dims(y_map, 0)
    spatial_map = np.concatenate((x_map, y_map), axis=0)

    spatial_map = torch.from_numpy(spatial_map)
    spatial_map = spatial_map.flatten(start_dim=1)
    spatial_map = spatial_map.swapaxes(0, 1)
    spatial_map = spatial_map.numpy()

    # convert to numpy
    latent_numpy = latent_tensor.numpy()

    # concatenate
    latent_numpy = np.concatenate((spatial_map, latent_numpy), axis=1)

    indicies = np.arange(0, label.shape[0])
    foreground_indicies = indicies[label == 1]
    background_indicies = indicies[label == 0]
    N = foreground_indicies.shape[0]

    # randomly sample from fg and bg indicies
    random_foreground = foreground_indicies[np.random.randint(0, high=foreground_indicies.shape[0], size=N)]
    random_background = background_indicies[np.random.randint(0, high=background_indicies.shape[0], size=N)]

    # concatenate
    idx = np.hstack((random_foreground, random_background))

    # reduce dimensionality with isomap
    #from sklearn.manifold import Isomap
    #embed = Isomap(n_components=2)
    from sklearn.manifold import TSNE
    embed = TSNE(n_components=2)
    latent_2d = embed.fit_transform(latent_numpy[idx, :])

    plt.clf()
    plt.scatter(latent_2d[:N, 0], latent_2d[:N, 1], marker='o', facecolors='none', edgecolors="#008b45", label="Pancreas", alpha=0.6)
    plt.scatter(latent_2d[N:, 0], latent_2d[N:, 1], marker='o', facecolors='none', edgecolors="#ffa07a", label="Background", alpha=0.6)
    plt.title("Dice score:  {0:.2f}".format(dice))
    plt.legend()
    plt.savefig(save_path)



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


def evaluate(test_loader, model_path, model_name, fold, ds_length):
    # Check if we have a GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load the model
    net = UNet(inChannels=1, outChannels=NUM_CHANNELS).to(device).double()

    # Make a new folder to store the output images
    try:
        os.mkdir(os.path.join(OUTPUT_DIR, MODEL_NAME.split(".")[0]))
    except:
        print("Output directory already exists")


    checkpoint = torch.load(os.path.join(model_path, model_name), map_location=torch.device(device))
    net.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    net.eval()

    # Read the config file
    config_file = model_name.split('.')[0] + "_config.pkl"
    f = open(os.path.join(model_path, config_file), 'rb')
    config_dict = pkl.load(f)
    f.close()

    PATCH_SIZE = config_dict["patch_size"]

    # Load the loss data
    loss_file = model_name.split('.')[0] + "_losses.pkl"
    f = open(os.path.join(model_path, loss_file), 'rb')
    [eps, av_train_error, av_train_dice, av_valid_error, av_valid_dice] = pkl.load(f)
    f.close()

    # Plot the loss curves and save
    plt.clf()
    plt.plot(eps, av_train_error, label="Train error")
    plt.plot(eps, av_valid_error, label="Valid error")
    plt.plot(eps, av_train_dice, label="Train Dice")
    plt.plot(eps, av_valid_dice, label="Valid Dice")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, MODEL_NAME.split(".")[0], "losses.png"))

    # load the filenames of test data
    #f = open(os.path.join(root_dir, "filenames_ts.pkl"), 'rb')
    #filenames_ts = pkl.load(f)
    #f.close()

    # create an empty array to store results
    dice_all = np.zeros((NUM_CHANNELS, ds_length))
    nsd_all = np.zeros((NUM_CHANNELS, ds_length))

    for j, (data, lab) in enumerate(test_loader):
        print("Evaluating test image {}".format(j))
        if PATCH_SIZE < 512:
            # Convert pytorch 3D image to a torchio 4d image
            subject = tio.Subject(image=tio.ScalarImage(tensor=data))

            grid_sampler = tio.inference.GridSampler(
                subject,
                (1, PATCH_SIZE, PATCH_SIZE),
                (0, PATCH_OVERLAP, PATCH_OVERLAP),
            )

            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')
            aggregator_latent = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')

            with torch.no_grad():
                for patches_batch in patch_loader:
                    # drop a dimension from the input tensor (dimension 2 is redundant)
                    input_tensor = torch.squeeze(patches_batch['image'][tio.DATA], dim=2)
                    locations = patches_batch[tio.LOCATION]
                    logits, latent = net(input_tensor.to(device).double())

                    # Expand 2nd dimension
                    logits = torch.unsqueeze(logits, dim=2)
                    latent = torch.unsqueeze(latent, dim=2)

                    aggregator.add_batch(logits, locations)
                    aggregator_latent.add_batch(latent, locations)

            output_logits = aggregator.get_output_tensor()
            latent_full = aggregator_latent.get_output_tensor()

            # drop redundant dimensions from output tensor, and one hot encode
            output_logits = torch.squeeze(output_logits)
            pred = output_logits.argmax(dim=0, keepdim=True).squeeze()
            one_hot = torch.FloatTensor(NUM_CHANNELS, 512, 512)
            one_hot.zero_()
            for i in range(NUM_CHANNELS):
                one_hot[i][pred == i] = 1
            one_hot = torch.unsqueeze(one_hot, dim=0)
        else:
            pred = net(data.to(device))  # shape (B, C, H, W)

            # convert preds to one-hot so it's comparable with output from nnU-Net
            max_idx = torch.argmax(pred, 1, keepdim=True)
            one_hot = torch.FloatTensor(pred.shape)
            one_hot.zero_()
            one_hot.scatter_(1, max_idx, 1)

        #dice = get_dice_per_class(one_hot, lab.to(device)).cpu().detach().numpy()
        #nsd = get_surface_dice(one_hot, lab.to(device), [1.5 for i in range(14)])

        # Fill Dice and NSD array
        #dice_all[:, j] = dice

        pred = one_hot.cpu().detach().numpy()
        lab = lab.cpu().detach().numpy()
        img = data.cpu().detach().numpy()

        # lose the first two image dimensions for plotting. Select only pancreas channel for evaluation
        pred = np.squeeze(pred)[1, :, :]
        lab = np.squeeze(lab)[1, :, :]
        img = np.squeeze(img)

        if True:
            # Visualise
            dice = PlotSliceAndPrediction(img, lab, pred, save_path=os.path.join(OUTPUT_DIR,
                                                                          MODEL_NAME.split(".")[0],
                                                                          "{}.png".format(j)))

            # Fill Dice and NSD array
            dice_all[:, j] = dice
            """
            try:
                visualiseLatent(latent_full, lab, dice, save_path=os.path.join(OUTPUT_DIR,
                                                                             "latent_tsne",
                                                                             "{}_spatial.png".format(j)))

            except:
                continue
                """

    # Save results
    f = open(os.path.join(OUTPUT_DIR, MODEL_NAME.split(".")[0], "results.pkl"), 'wb')
    pkl.dump([dice_all, nsd_all], f)
    f.close()

    print('Done')






def main():
    test_loader, ds_length = create_test_dataset(DATA_DIR, NUM_CHANNELS)
    evaluate(test_loader, MODEL_DIR, MODEL_NAME, FOLD, ds_length)


if __name__ == "__main__":
    main()