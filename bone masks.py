# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:35:38 2021

@author: hermi
"""
#bone masks rip
#%%
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage
import cv2
window = 350
level = 50
#%%
def GetSliceNumber(segment):
  slice_number = 0
  max_range = len(sitk.GetArrayFromImage(segment))
  for x in range(0,max_range):
    seg_slice_2 = sitk.GetArrayFromImage(segment)[x,:,:]
    val = np.sum(seg_slice_2)
    if val != 0:
      slice_number = x
  return slice_number

def GetSliceNumber2(segment):
  slice_numbers = []
  max_range = len(sitk.GetArrayFromImage(segment))
  for x in range(0,max_range):
    seg_slice_2 = sitk.GetArrayFromImage(segment)[x,:,:]
    val = np.sum(seg_slice_2)
    if val != 0:
      slice_numbers.append(x)
  return slice_numbers

def PrintTrainingDataLiv(slicef_array, maskf_array, idf_array, i ):
  print("HNSCC Patient " + str(idf_array[i]), "index: ", i)
  plt.imshow(slicef_array[i], cmap = plt.cm.gray, vmin = level/2 - window, vmax = level/2 + window)
  seg_slice = maskf_array[i]
  seg_slice = seg_slice.astype(float)
  seg_slice[seg_slice == 0] = np.nan
  plt.imshow(seg_slice, cmap = plt.cm.autumn, alpha = 0.6)
  plt.show()
#%%
def extract_bone_masks(dcm_array, slice_number, threshold=200, radius=2, worldmatch=False):
    """
    Calculate 3D bone mask and remove from prediction. 
    Used to account for partial volume effect.
​
    Args:
        dcm_array - 3D volume
        slice number - slice where segmentation being performed
        threshold (in HU) - threshold above which values are set to 1 
        radius (in mm) - Kernel radius for binary dilation
    """
    #img = sitk.GetImageFromArray(dcm_array)
    img = dcm_array
    # Worldmatch tax
    if worldmatch:
        img -= 1024
    # Apply threshold
    bin_filt = sitk.BinaryThresholdImageFilter()
    bin_filt.SetOutsideValue(1)
    bin_filt.SetInsideValue(0)
    bin_filt.SetLowerThreshold(-1024)
    bin_filt.SetUpperThreshold(threshold)
    bone_mask = bin_filt.Execute(img)
    pix = bone_mask.GetSpacing()
    # Convert to pixels
    pix_rad = [int(radius//elem) for elem in pix]
    # Dilate mask
    dil = sitk.BinaryDilateImageFilter()
    dil.SetKernelType(sitk.sitkBall)
    dil.SetKernelRadius(pix_rad)
    dil.SetForegroundValue(1)
    dilated_mask = dil.Execute(bone_mask)
    
    return sitk.GetArrayFromImage(dilated_mask)[slice_number]
#%%
def crop(slices_arr, masks_arr, bone_arr, sans_le_bone_arr):
  #taking a mean of the centre of image and centre of intensity
  slices_cropped = []
  masks_cropped = []
  bone_cropped = []
  mask_sans_le_bone_cropped = []
  for i in range(0,len(slices_arr)):
      crop_slice = slices_arr[i][0:250, 0:512]
      ret,threshold = cv2.threshold(crop_slice,200,250, cv2.THRESH_TOZERO)
      coords = ndimage.measurements.center_of_mass(threshold)
      size = 130
      x_min = int(((coords[0] - size)+126)/2)
      x_max = int(((coords[0] + size)+386)/2)
      y_min = int(((coords[1] - size)+126)/2)
      y_max = int(((coords[1] + size)+386)/2)
      #x_min = 126
      #x_max = 386
      #y_min = 126
      #y_max = 386
      crop_image = slices_arr[i][x_min:x_max, y_min:y_max]
      crop_seg = masks_arr[i][x_min:x_max, y_min:y_max]
      crop_bone = bone_arr[i][x_min:x_max, y_min:y_max]
      crop_sans = sans_le_bone_arr[i][x_min:x_max, y_min:y_max]
      slices_cropped.append(crop_image)
      masks_cropped.append(crop_seg)
      bone_cropped.append(crop_bone)
      mask_sans_le_bone_cropped.append(crop_sans)
  return np.asarray(slices_cropped), np.asarray(masks_cropped), np.asarray(bone_cropped), np.asarray(mask_sans_le_bone_cropped)

#%%
slices_array = []
mask_array = []
id_array = []
pixel_areas = []
masks_sans_le_bone = []
bone_masks = []
#path = "C:\Users\hermi\OneDrive\Documents\physics year 4\Mphys\L3 scans\My_segs\P"
#os.getcwd()
#path = 'C:\\Users\\hermi\\OneDrive\\Documents\\physics year 4\\Mphys\\L3_scans\\My_segs\\P'
path = 'C:\\Users\\hermi\\OneDrive\\Documents\\physics year 4\\Mphys\\L3_scans\\My_segs\\P'
#path = 'C:\\Users\\hermi\\OneDrive\\Documents\\physics year 4\\Mphys\\L3_scans\\plusminussegs\\P'

#i =1
#path_ct = path + str(i) + "_RT_sim_ct.nii"
#path_seg = path + str(i) + "_RT_sim_seg.nii.gz"
#ct_scan = sitk.ReadImage(path_ct)
#segment = sitk.ReadImage(path_seg)
#slice_no = GetSliceNumber(segment)
#ct_slice = sitk.GetArrayFromImage(ct_scan)[slice_no,:,:]
#ct_slice = ct_slice.astype(float)
#
#mask = sitk.GetArrayFromImage(segment)[slice_no,:,:]
#mask = mask.astype(float)
#
#boneMask = np.logical_not(extract_bone_masks(ct_scan, slice_no))
##bone_masks.append(boneMask)
#print(boneMask.shape)
#plt.imshow(boneMask, cmap = "gray")
#%%
def ReadInAndSave(TopOfRange, skip):
    slices_array = []
    mask_array = []
    id_array = []
    pixel_areas = []
    masks_sans_le_bone = []
    bone_masks = []
    for i in range(1,TopOfRange):
      if i not in skip:
        path_ct = path + str(i) + "_RT_sim_ct.nii"
        path_seg = path + str(i) + "_RT_sim_seg.nii.gz"
        ct_scan = sitk.ReadImage(path_ct)
        segment = sitk.ReadImage(path_seg)
        slice_no = GetSliceNumber(segment)
        pixel_areas.append(((ct_scan.GetSpacing())[0])*((ct_scan.GetSpacing())[1]))
        ct_slice = sitk.GetArrayFromImage(ct_scan)[slice_no,:,:]
        ct_slice = ct_slice.astype(float)
        slices_array.append(ct_slice)
        mask = sitk.GetArrayFromImage(segment)[slice_no,:,:]
        mask = mask.astype(float)
        #mask[mask_array == 0] = np.nan
        mask_array.append(mask)
        id_array_element = "01-00" + str(i)
        id_array.append(id_array_element)
        #boneMask = np.logical_not(extract_bone_masks(ct_scan, slice_no))
        #bone_masks.append(boneMask)
        #masks_sans_le_bone.append(np.logical_and(mask, boneMask))
    return slices_array, mask_array, id_array, pixel_areas, masks_sans_le_bone, bone_masks

skip = [24, 25, 37]
slices_array, mask_array, id_array, pixel_areas, masks_sans_le_bone, bone_masks = ReadInAndSave(39, skip)
print(id_array)
#%%
for i in range(1,39):
  if i !=24 and i !=25 and i != 37:
    path_ct = path + str(i) + "_RT_sim_ct.nii"
    path_seg = path + str(i) + "_RT_sim_seg.nii.gz"
    ct_scan = sitk.ReadImage(path_ct)
    segment = sitk.ReadImage(path_seg)
    slice_no = GetSliceNumber(segment)
    pixel_areas.append(((ct_scan.GetSpacing())[0])*((ct_scan.GetSpacing())[1]))
    ct_slice = sitk.GetArrayFromImage(ct_scan)[slice_no,:,:]
    ct_slice = ct_slice.astype(float)
    slices_array.append(ct_slice)
    mask = sitk.GetArrayFromImage(segment)[slice_no,:,:]
    mask = mask.astype(float)
    #mask[mask_array == 0] = np.nan
    mask_array.append(mask)
    id_array_element = "01-00" + str(i)
    id_array.append(id_array_element)
    boneMask = np.logical_not(extract_bone_masks(ct_scan, slice_no))
    bone_masks.append(boneMask)
    masks_sans_le_bone.append(np.logical_and(mask, boneMask))
  
#%%
for i in range(90,176):
  if i != 100 and i != 114 and i != 116 and i != 136 and i != 137 and i != 143 and i != 156 and i != 157 and i != 169:
    path_ct = path2 + str(i) + ".nii.gz"
    path_seg = path2 + str(i) + "_seg.nii.gz"
    ct_scan = sitk.ReadImage(path_ct)
    segment = sitk.ReadImage(path_seg)
    slice_no = GetSliceNumber(segment)
    pixel_areas.append(((ct_scan.GetSpacing())[0])*((ct_scan.GetSpacing())[1]))
    ct_slice = sitk.GetArrayFromImage(ct_scan)[slice_no,:,:]
    ct_slice = ct_slice.astype(float)
    slices_array.append(ct_slice)
    mask = sitk.GetArrayFromImage(segment)[slice_no,:,:]
    mask = mask.astype(float)
    #mask[mask_array == 0] = np.nan
    mask_array.append(mask)
    id_array_element = "01-00" + str(i)
    id_array.append(id_array_element)
    boneMask = np.logical_not(extract_bone_masks(ct_scan, slice_no))
    bone_masks.append(boneMask)
    masks_sans_le_bone.append(np.logical_and(mask, boneMask))

#%%
for i in range(1,6):
   path_ct = path + str(i) + "_ct.nii"
   path_seg = path + str(i) + "_segs.nii.gz"
   ct_scan = sitk.ReadImage(path_ct)
   segment = sitk.ReadImage(path_seg)
   slice_nos = GetSliceNumber2(segment)
   print(slice_nos)
   pixel_areas.append(((ct_scan.GetSpacing())[0])*((ct_scan.GetSpacing())[1]))
   for z in range(0, len(slice_nos)):
       ct_slice = sitk.GetArrayFromImage(ct_scan)[slice_nos[z],:,:]
       ct_slice = ct_slice.astype(float)
       slices_array.append(ct_slice)
       mask = sitk.GetArrayFromImage(segment)[slice_nos[z],:,:]
       mask = mask.astype(float)
       #mask[mask_array == 0] = np.nan
       mask_array.append(mask)
       boneMask = np.logical_not(extract_bone_masks(ct_scan, slice_nos[z]))
       bone_masks.append(boneMask)
       masks_sans_le_bone.append(np.logical_and(mask, boneMask))
       id_array_element = "01-00" + str(i) + ", " + str(slice_nos[z])
       id_array.append(id_array_element)
     
#%%
slices_array = np.asarray(slices_array)
masks_array = np.asarray(mask_array)
id_array = np.asarray(id_array)
area_array = np.asarray(pixel_areas)
bone_array = np.asarray(bone_masks)
mask_sans_le_bone = np.asarray(masks_sans_le_bone)#only for training data
#%%
print(slices_array.shape, bone_array.shape)
bone_array[bone_array == True] == 1
bone_array[bone_array == False] == 0


for i in range(0,1):
    print(id_array[i], "index:", i)
    plt.imshow(slices_array[i], cmap = "gray")
    masks_array[masks_array[i] == 0] == np.nan
    plt.imshow(masks_array[i], cmap = "autumn", alpha = 0.5)
    #plt.imshow(bone_array[i], cmap = "cool", alpha = 0.5)#does not like 0 to nan
    plt.show()

#%%
slices_cropped, masks_cropped, bone_cropped, masks_slb_crop = crop(slices_array, masks_array, bone_masks, mask_sans_le_bone)

print(slices_cropped.shape, masks_slb_crop.shape)
#%%
for i in range(0,77):
    print(id_array[i], "index:", i)
    plt.imshow(slices_cropped[i], cmap = "gray")
    plt.imshow(masks_cropped[i], cmap = "autumn", alpha = 0.5)
    plt.imshow(bone_cropped[i], cmap = "cool", alpha = 0.5)#does not like 0 to nan
    plt.show()
  
#%%
pathy = 'C:\\Users\\hermi\\OneDrive\\Documents\\physics year 4\\Mphys\\L3_scans\\My_segs\\training_data_avec_bone'
training_data = np.savez(pathy, slices = slices_cropped, masks = masks_cropped, ids=id_array, pixel_areas = area_array, bone_masks = bone_cropped, boneless_masks = masks_slb_crop)
#%%
pathy = 'C:\\Users\\hermi\\OneDrive\\Documents\\physics year 4\\Mphys\\L3_scans\\My_segs\\c3s_plusminus_3'
c3s_plusminus_3 = np.savez(pathy, slices = slices_cropped, masks = masks_cropped, ids=id_array, pixel_areas = area_array, bone_masks = bone_cropped, boneless_masks = masks_slb_crop)

