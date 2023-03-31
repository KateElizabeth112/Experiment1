# Scipts for plotting results
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os


ROOT_DIR = '/Users/katecevora/Documents/PhD'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'images/test')


def main():
    models = ["nnunet", "unet_v2_1",  "unet_v2_3", "unet_v2_4", "unet_v2_5"]
    names = ["nnunet", "random, (50, 99.5)",  "weighted, (50, 99.5)", "weighted (2.5, 99.5)", "random (2.5, 99.5)"]
    colors = ["#ffa07a", "#008b45", "#ed90e1", "#00ced1", "#fa8072", "#15f4ee"]

    plt.clf()

    for j in range(len(models)):
        # open the results for the model
        f = open(os.path.join(OUTPUT_DIR, models[j], "results.pkl"), 'rb')
        [dice_all, nsd_all] = pkl.load(f)
        f.close()

        res = dice_all[1, :]
        y = np.zeros(res.shape)
        y.fill(j+1)

        plt.scatter(res, y, label="Mean: {0:.2f}, Std.: {1:.2f}".format(np.mean(res), np.std(res)), marker='o', facecolors='none', edgecolors=colors[j])
        plt.scatter(np.mean(res), j+1, marker='x', color='k', s=80)

    plt.xlabel("Dice Score")
    plt.yticks(range(1, j+2), names)
    plt.ylim([0, j+2])
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()