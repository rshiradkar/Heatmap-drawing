import os
import numpy as np
import nibabel as nib
import nrrd
from radiomics import featureextractor, getFeatureClasses
import SimpleITK as sitk
import six  # Ensure you have the 'six' library installed for Python 3 compatibility
import seaborn as sns
import pandas as pd
import scipy.ndimage
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap

dir_root = 'C:/Users/rshirad/OneDrive - Emory University/Augusta_PCAa_CVD/Dataset/'

# cortext_norm_output_path = '/Users/IyerFamMac/Desktop/opencv_project/'
# cortext_MinMax_norm_output_path = '/Users/IyerFamMac/Desktop/opencv_project/'

subdirectories = [f for f in os.listdir(dir_root) if os.path.isdir(os.path.join(dir_root, f))]
# print('folder_names shape: ', len(subdirectories))
# print('folder_names: ', subdirectories)


# for folder_name in subdirectories:
folder_name = subdirectories[0]
data_dir = dir_root + folder_name

for filename in os.listdir(data_dir):
    if filename.endswith('.seg.nrrd'):
        mask_path = data_dir + '/' + filename
        print('mask_path: ', mask_path)
        
    if filename.endswith('equalized.nii.gz'): # Filter the list to include only .nii and .nii.gz files
        mri_path = data_dir + '/' + filename
        print('mri_path: ', mri_path)

imageName = sitk.ReadImage(mri_path)
image_array = sitk.GetArrayFromImage(imageName)
# image_array = imageName.get_fdata()
print('image_array shape: ', image_array.shape)

mask = sitk.ReadImage(mask_path)
mask_array = sitk.GetArrayFromImage(mask)
print('image_array shape: ', mask_array.shape)


#align mask with volumn
# Get information from the image
image_spacing = imageName.GetSpacing()
image_origin = imageName.GetOrigin()
image_direction = imageName.GetDirection()

# Create a resampling object
resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing(image_spacing)
resampler.SetOutputOrigin(image_origin)
resampler.SetOutputDirection(image_direction)
resampler.SetSize(imageName.GetSize())

# Resample the mask
aligned_mask = resampler.Execute(mask)
aligned_mask_array = sitk.GetArrayFromImage(aligned_mask)

# Get unique values in the msk array
unique_values = np.unique(aligned_mask_array)

# print(unique_values)

extractor = featureextractor.RadiomicsFeatureExtractor('params.yaml')
features = extractor.execute(imageName, aligned_mask, voxelBased=True)
print('features keys length: ', len(features)) 


#generate heatmaps for all slice where there is a mask. (sum_mask is >0 )
# Select slices where the mask value is not all zero
non_zero_slices = np.any(aligned_mask_array != 0, axis=(1, 2))

# Get the indices of non-zero slices
non_zero_slice_indices = np.where(non_zero_slices)[0]

feature_list = ['original_ngtdm_Busyness', 'original_glcm_ClusterShade', 'original_firstorder_Skewness', 'original_firstorder_Maximum', 'original_glszm_GrayLevelNonUniformityNormalized']
for key, val in six.iteritems(features):
    print('key: ', key)
    print('key type: ', type(key))
    
    #if isinstance(val, sitk.Image) and key in feature_split_results.values: 
    if isinstance(val, sitk.Image): 
        if key in feature_list: 
            print('Feature found!!!!!')
            print('key: ', key)

            ###part 1: resampling the featurmap as the mask shape!!!!!!!!!!!!!!!!!!!!
            rif = sitk.ResampleImageFilter()
            #rif.SetInterpolator(sitk.sitkNearestNeighbor)
            #rif.SetInterpolator(sitk.sitkLinear)
            rif.SetInterpolator(sitk.sitkBSpline)
            rif.SetReferenceImage(aligned_mask)
            val_new = rif.Execute(val)

            # Convert the SimpleITK image to a NumPy array
            parametermap = sitk.GetArrayFromImage(val_new)
            print('parametermap shape: ', parametermap.shape)
            # Check for NaN values
            if np.isnan(parametermap).any():
                print("Image contains NaN values.")
                
            # Display overlay for each non-zero slice
            for slice_index in non_zero_slice_indices:
                # Get the corresponding slice in the MRI image
                #####part 2: Min_Max scaler the features###################
    #                     # Flatten the 3D array
    #                     flattened_array = parametermap.flatten()

    #                     # Apply MinMaxScaler
    #                     scaler = MinMaxScaler()
    #                     scaled_array = scaler.fit_transform(flattened_array.reshape(-1, 1))
    #                     # Reshape the scaled array back to the original shape
    #                     scaled_array = scaled_array.reshape(parametermap.shape)
    #                     print('scaled_array shape: ', scaled_array.shape)

    #                     heatmap_data = scaled_array[slice_index, :, :]
    #                     print('heatmap_data shape: ', heatmap_data.shape)

                heatmap_data = parametermap[slice_index, :, :]
                print('heatmap_data shape: ', heatmap_data.shape)

                #########part 3:plot the heatmap and overlay features#########
                # Get the corresponding slice in the mask
                masks = aligned_mask_array[slice_index, :, :]  # Assuming you have 'aligned_mask_array' available
                print('mask slice shape: ', masks.shape)
                print('heatmap slice shape: ', heatmap_data.shape)

                # Set the area of the heatmap within the mask as non or blank
                #heatmap_data = np.where(masks == 1, heatmap_data, 0)
                heatmap_data = np.where(masks == 1, heatmap_data, np.nan)
                # Set the pixels of heatmap_data to None where the mask equals 0
                heatmap_data[masks == 0] = np.nan

                ####align the mask shape with heatmap_data shape
                print(masks[masks==1].shape, heatmap_data[heatmap_data!=0].shape)

                # Plot a dummy image to create the color bar
                # Extract 'Min' and 'Max' values for the specified feature
                # vmin = minmax_df.loc[minmax_df['Feature'] == key, 'Min'].values[0]
                # vmax = minmax_df.loc[minmax_df['Feature'] == key, 'Max'].values[0]
                # Create a new figure and axis for each iteration
                fig, ax = plt.subplots()
                ax.imshow(image_array[slice_index, :, :], cmap='gray')
                

                # Choose the 'turbo' colormap
                turbo_cmap = plt.get_cmap('turbo')

                # Plot the MRI image and overlay the heatmap with fixed colors, range, and transparency
                # im = ax.imshow(heatmap_data, cmap=turbo_cmap, alpha=0.9, vmin=vmin, vmax=vmax)
                im = ax.imshow(heatmap_data, cmap=turbo_cmap, alpha=0.9)

                # Add title and colorbar
                plt.title(str(key) + ' Overlay on Heatmap')
                cbar = plt.colorbar(im, orientation='vertical', ax=ax)
                

                # Save the plot to a file (e.g., PNG)
            
                output_file_path = 'C:/Users/rshirad/OneDrive - Emory University/Augusta_PCAa_CVD/heatmaps/'
                plt.savefig(output_file_path + str(folder_name) + "_" + str(slice_index) + "_" + str(key) + ".png")
                plt.close()  # Close the plot to avoid displaying multiple plots
                # Show the plot (or save, etc.)
                plt.show()
                
                print(f"Heatmap saved to: {output_file_path}")  

