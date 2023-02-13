# Prepare data for 2D nnUNet
import numpy as np
import os
import nibabel as nib

root_dir = '/vol/biomedic3/kc2322/'

data_dir = os.path.join(root_dir, 'data/MSDPancreas/MSDPancreas/imagesTrMIRTK')
img_dir = os.path.join(root_dir, 'data/MSDPancreas/MSDPancreas/nnUNet/imagesTr')
lab_dir = os.path.join(root_dir, 'data/MSDPancreas/MSDPancreas/nnUNet/labelsTr')


def prep2DData():
    # list the sub-folders in the data directory
    folders = os.listdir(data_dir)

    for f in folders:
        if f.startswith("pancreas"):
            try:
                print(f)

                # open label and image
                img_nii = nib.load(os.path.join(data_dir, f, f + "_warp.nii.gz"))
                gt_nii = nib.load(os.path.join(data_dir, f, f + "_label_warp.nii.gz"))

                gt = gt_nii.get_fdata()

                # flatten label and find extent
                gt_flat = np.sum(gt, axis=(0, 1))
                indicies = np.where(gt_flat > 0)[0]

                # Find where extent of pancreas slice is greatest
                x = np.argmax(gt_flat)

                # save the slices around the middle (along wih labels)
                for i in range(3):
                    slice_idx = x - i
                    gt_slice = gt_nii.slicer[:, :, slice_idx:slice_idx + 1]
                    img_slice = img_nii.slicer[:, :, slice_idx:slice_idx + 1]

                    nib.save(gt_slice, os.path.join(lab_dir, f + "{0}.nii.gz".format(slice_idx)))
                    nib.save(img_slice, os.path.join(img_dir, f + "{0}_0000.nii.gz".format(slice_idx)))

                for i in range(1, 3):
                    slice_idx = x + i
                    gt_slice = gt_nii.slicer[:, :, slice_idx:slice_idx + 1]
                    img_slice = img_nii.slicer[:, :, slice_idx:slice_idx + 1]

                    nib.save(gt_slice, os.path.join(lab_dir, f + "{0}.nii.gz".format(slice_idx)))
                    nib.save(img_slice, os.path.join(img_dir, f + "{0}_0000.nii.gz".format(slice_idx)))
            except:
                print(f + " failed")
                continue


def main():
    prep2DData()


if __name__ == "__main__":
    main()

