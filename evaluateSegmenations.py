import SimpleITK as sitk
from display import DisplayRegistration2D
import os

# check MIRTK registrations by creating plots

data_dir = "/vol/biomedic3/kc2322/data/MSDPancreas/MSDPancreas/imagesTr/"
reg_dir = "/vol/biomedic3/kc2322/data/MSDPancreas/MSDPancreas/imagesTrMIRTK/"
img_dir = "/vol/biomedic3/kc2322/images/MIRTKRegistrations"

target = "pancreas_376.nii.gz"


def main():
    files = os.listdir(data_dir)
    target_img = sitk.ReadImage(os.path.join(data_dir, target), sitk.sitkFloat32)

    for f in files:
        if f.endswith(".gz"):
            name = f.split('.')[0]
            print(name)

            # load original and warped images
            source_img = sitk.ReadImage(os.path.join(data_dir, f), sitk.sitkFloat32)
            warped_img = sitk.ReadImage(os.path.join(reg_dir, name, name+"_warp.nii.gz"), sitk.sitkFloat32)

            vox_source = source_img.GetSpacing()

            DisplayRegistration2D(sitk.GetArrayFromImage(target_img), sitk.GetArrayFromImage(source_img),
                                  sitk.GetArrayFromImage(warped_img), vox_source,
                                  save_path=os.path.join(img_dir, name + ".png"))


if __name__ == "__main__":
    main()