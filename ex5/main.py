import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from skimage import io


def resize_image_itk(itk_image, new_size, re_sample_method=sitk.sitkNearestNeighbor):
    re_sampler = sitk.ResampleImageFilter()
    origin_size = itk_image.GetSize()  # 原来的体素块尺寸
    origin_spacing = itk_image.GetSpacing()
    new_size = np.array(new_size, float)
    factor = origin_size / new_size
    new_spacing = origin_spacing * factor
    new_size = new_size.astype(np.int)  # spacing肯定不能是整数
    re_sampler.SetReferenceImage(itk_image)  # 需要重新采样的目标图像
    re_sampler.SetSize(new_size.tolist())
    re_sampler.SetOutputSpacing(new_spacing.tolist())
    re_sampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    re_sampler.SetInterpolator(re_sample_method)
    itk_img_re_sampled = re_sampler.Execute(itk_image)  # 得到重新采样后的图像
    return itk_img_re_sampled


def mapping_hu_to_pixel(hu_img, wl, ws):
    ct_min = wl - ws / 2
    ct_max = wl + ws / 2

    hu_img[hu_img <= ct_min] = ct_min
    hu_img[hu_img >= ct_max] = ct_max

    return (((hu_img - ct_min) / (ct_max - ct_min)) * 255).astype(np.uint8)


reader = sitk.ImageFileReader()
reader.SetImageIO("NiftiImageIO")
reader.SetFileName("./COVID19_A_1_1.nii.gz")
image = reader.Execute()
image_re_sampled = resize_image_itk(image, (512, 512, 512))
img_hu = sitk.GetArrayFromImage(image_re_sampled)
img = mapping_hu_to_pixel(img_hu, -645, 4756)

x, y, z = img.shape
for i in range(x):
    io.imsave("./output/transverse_plane/transverse_plane" + str(i) + ".png", img[i, :, :])

for i in range(y):
    io.imsave("./output/sagittal_plane/sagittal_plane" + str(i) + ".png", img[:, i, :])

for i in range(z):
    io.imsave("./output/coronal_plane/coronal_plane" + str(i) + ".png", img[:, :, i])

