import pickle as pkl
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import time

from registration import RigidRegistration
from display import Display3D, DisplayRegistration2D, Display2D, DisplayOverlay2D

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


def main():
    files = os.listdir(data_dir)

    reference_img_name = "pancreas_376.nii.gz"
    #target_img_name = "pancreas_392.nii.gz"

    sitk_reference_img = sitk.ReadImage(os.path.join(data_dir, reference_img_name), sitk.sitkFloat32)

    for f in files:
        if f.endswith(".gz"):
            name = f.split(".")[0]
            print(name)

            start_time = time.time()

            sitk_target_img = sitk.ReadImage(os.path.join(data_dir, f), sitk.sitkFloat32)

            sitk_warped_image = RigidRegistration(sitk_reference_img, sitk_target_img)

            vox_target = sitk_target_img.GetSpacing()

            DisplayRegistration2D(sitk.GetArrayFromImage(sitk_reference_img), sitk.GetArrayFromImage(sitk_target_img),
                                  sitk.GetArrayFromImage(sitk_warped_image), vox_target,
                                  save_path=os.path.join('images/registrations', name + ".png"))

            end_time = time.time()

            print(end_time - start_time)


if __name__ == "__main__":
    main()

