import SimpleITK as sitk


# SimpleITK rigid registration with Simplex optimization
def RigidRegistration(target_image, source_image):
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.1)

    # Image interpolation.
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsAmoeba(100.0, 500)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4])

    # Spatial transformation model.
    initial_transform = sitk.CenteredTransformInitializer(target_image, source_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Run the registration.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(target_image, source_image)

    # Generate warped image
    warped_image = sitk.Resample(source_image, target_image, final_transform, sitk.sitkLinear, 0.0, source_image.GetPixelID())

    return warped_image