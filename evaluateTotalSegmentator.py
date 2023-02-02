import os
import numpy as np
import nibabel as nib

SYSTEM = 'remote'

if SYSTEM == 'local':
    root_dir = '/Users/katecevora/Documents/PhD'
    os.chdir(root_dir)

    data_dir = 'data/MSDPancreas/ImagesTr'
    labels_dir = 'data/MSDPancreas/LabelsTr'
    preds_dir = 'data/MSDPancreas/MSDPancreas/TotalSegmentator'
else:
    root_dir = '/vol/biomedic3/kc2322/'

    data_dir = os.path.join(root_dir, 'data/MSDPancreas/MSDPancreas/imagesTr')
    labels_dir = os.path.join(root_dir, 'data/MSDPancreas/MSDPancreas/labelsTr')
    preds_dir = os.path.join(root_dir, 'data/MSDPancreas/MSDPancreas/TotalSegmentator')


def getDiceScores():
    # list the files in the training folder
    files = os.listdir(data_dir)
    scores = {}

    for f in files:
        if f.endswith(".gz"):
            name = f.split(".")[0]

            img = nib.load(os.path.join(data_dir, f))
            gt = nib.load(os.path.join(labels_dir, f))
            pred = nib.load(os.path.join(preds_dir, name, "pancreas.nii.gz"))

            header = img.header
            vox_size = header.get_zooms()
            img_raw = img.get_fdata()
            lab_raw = pred.get_fdata()
            gt_raw = gt.get_fdata()

            #DisplayOverlay2D(img_raw, lab_raw, vox_size, plane=2,
            #                 save_path=os.path.join("images", "totalsegmentator.png"))

            # Calculate Dice score
            gt_raw[gt_raw == 2] = 0
            dice = np.sum(lab_raw[gt_raw == 1]) * 2.0 / (np.sum(lab_raw) + np.sum(gt_raw))

            # Add Dice score to a dictionary
            scores[name] = dice

            return scores


def main():

    scores = getDiceScores()
    print("Average Dice score = {0:.2f}".format(sum(scores.values()) / len(scores)))


if __name__ == "__main__":
    main()
