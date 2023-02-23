import pickle as pkl
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import time

from registration import RigidRegistration2
from display import Display3D, DisplayRegistration2D, Display2D, DisplayOverlay2D, PlotSliceAndPrediction

root_dir = '/Users/katecevora/Documents/PhD'
os.chdir(root_dir)

data_dir = 'data/MSDPancreas/ImagesTr'
labels_dir = 'data/MSDPancreas/LabelsTr'


# Iterate over the dataset and save images along the central slice of each axis with labels
def VisualiseDataset(files):
    for f in files:
        if f.endswith(".gz"):
            name = f.split('.')[0]
            print(name)

            img_ct_nii = nib.load(os.path.join(data_dir, f))
            header = img_ct_nii.header
            vox_size = header.get_zooms()
            img_raw = img_ct_nii.get_fdata()

            # load the image labels
            lab_nii = nib.load(os.path.join(labels_dir, f))
            lab_raw = lab_nii.get_fdata()

            DisplayOverlay2D(img_raw, lab_raw, vox_size, plane=0,
                             save_path=os.path.join("images/sagittal", name + ".png"))
            DisplayOverlay2D(img_raw, lab_raw, vox_size, plane=1,
                             save_path=os.path.join("images/coronal", name + ".png"))
            DisplayOverlay2D(img_raw, lab_raw, vox_size, plane=2,
                             save_path=os.path.join("images/longitudinal", name + ".png"))


def InspectTotalSegmentator():
    img = nib.load(os.path.join(data_dir, "pancreas_001.nii.gz"))
    gt = nib.load(os.path.join(labels_dir, "pancreas_001.nii.gz"))
    lab = nib.load("pancreas.nii.gz")

    header = img.header
    vox_size = header.get_zooms()
    img_raw = img.get_fdata()
    lab_raw = lab.get_fdata()
    gt_raw = gt.get_fdata()

    DisplayOverlay2D(img_raw, lab_raw, vox_size, plane=2,
                     save_path=os.path.join("images", "totalsegmentator.png"))

    # Calculate Dice score
    k = 1
    gt_raw[gt_raw == 2] = 0
    dice = np.sum(lab_raw[gt_raw == k]) * 2.0 / (np.sum(lab_raw) + np.sum(gt_raw))

    print(dice)


def RegisterImages(reference_img_name):
    files = os.listdir(data_dir)
    sitk_reference_img = sitk.ReadImage(os.path.join(data_dir, reference_img_name), sitk.sitkFloat32)

    for f in files:
        if f.endswith(".gz"):
            name = f.split(".")[0]
            print(name)

            start_time = time.time()

            sitk_moving_img = sitk.ReadImage(os.path.join(data_dir, f), sitk.sitkFloat32)

            sitk_warped_image = RigidRegistration2(sitk_reference_img, sitk_moving_img)


            vox_moving = sitk_moving_img.GetSpacing()

            DisplayRegistration2D(sitk.GetArrayFromImage(sitk_reference_img), sitk.GetArrayFromImage(sitk_moving_img),
                                  sitk.GetArrayFromImage(sitk_warped_image), vox_moving,
                                  save_path=os.path.join('images/registrations2', name + ".png"))

            end_time = time.time()

            print(end_time - start_time)




def main():
    img_path = os.path.join(root_dir, "data/MSDPancreas/2D/imagesTr/")
    label_path = os.path.join(root_dir, "data/MSDPancreas/2D/labelsTr/")
    pred_path = os.path.join(root_dir, "data/MSDPancreas/2D/inference/")
    output_dir = os.path.join(root_dir, "images/2D/central_5_slices_registered/")
    files = os.listdir(img_path)
    for f in files:
        # try to load the file and the label so we can visualise them

        # extract the file name so we can also open the label file
        id = f.split('_')[1]
        label_name = "pancreas_" + id + ".nii.gz"
        print(label_name, f)

        img_nii = nib.load(os.path.join(img_path, f))
        lab_nii = nib.load(os.path.join(label_path, label_name))
        pred_nii = nib.load(os.path.join(pred_path, label_name))

        # Visualise
        PlotSliceAndPrediction(np.rot90(img_nii.get_fdata()[:, :, 0]), np.rot90(lab_nii.get_fdata()[:, :, 0]),
                               np.rot90(pred_nii.get_fdata()[:, :, 0]),
                               save_path=os.path.join(output_dir, "pancreas_" + id + ".png"))




    print("Done")


if __name__ == "__main__":
    main()

