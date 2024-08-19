from astropy.io import fits
import numpy as np
from scipy.ndimage import label, find_objects
from PIL import Image
import pandas as pd
from multiprocessing import Pool
import warnings
import math
import os

warnings.filterwarnings("ignore")

def preprocess(AR_patch, lb, ub):
    Denoised_AR_patch = np.clip(AR_patch, -256, 255)

    noise_removal_mask = np.logical_and(Denoised_AR_patch >= lb, Denoised_AR_patch <= ub)

    Denoised_AR_patch[noise_removal_mask] = 0.

    return Denoised_AR_patch


def bytscal_with_nan(input_array, min_value=None, max_value=None):
    if min_value is None:
        min_value = np.nanmin(input_array)  # Use np.nanmin to ignore NaN values
    if max_value is None:
        max_value = np.nanmax(input_array)  # Use np.nanmax to ignore NaN values
 
    # Perform byte scaling while preserving NaN values
    scaled_array = np.where(np.isnan(input_array), np.nan, ((input_array - min_value) / (max_value - min_value) * 255).astype(np.uint8))

    return scaled_array


def bitmap_cropping(AR_Patch, bitmap_data):
    # Check for NaN values in the bitmap data
    nan_mask = np.isnan(bitmap_data)

    # Threshold the bitmap data to have 1s and 0s
    bitmap_data = (bitmap_data > 2).astype(np.uint8)

    # Exclude regions with NaN values from the binary bitmap
    bitmap_data[nan_mask] = 0

    # Label the connected components in the updated bitmap
    labels, num_features = label(bitmap_data)

    # Calculate the sizes of each region
    component_sizes = np.bincount(labels.ravel())

    # Find the label of the largest component
    largest_component_label = np.argmax(component_sizes[1:]) + 1

    # Find the coordinates of the largest component
    largest_component_coords = np.argwhere(labels == largest_component_label)

    # Get the minimum and maximum coordinates of the largest component
    min_x, min_y = largest_component_coords.min(axis=0)
    max_x, max_y = largest_component_coords.max(axis=0)

    # Crop the magnetogram to the region of interest
    cropped_magnetogram = AR_Patch[min_x:max_x + 1, min_y:max_y + 1]
    return cropped_magnetogram

def padding(X, constant_value=0):
    h,w = X.shape
    pad_h = 512 - h
    pad_w = 512 - w
    
    if w % 2 != 0 and h % 2 != 0:
        pad_top = math.floor(pad_h/2)
        pad_bottom = pad_top+1
        pad_left = math.floor(pad_w/2)
        pad_right = pad_left+1
        X = np.pad(X, ((int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))), 
                    'constant',constant_values = constant_value) #top bottom left right 

    elif w % 2 != 0 and h % 2 == 0:
        pad_top = math.floor(pad_h/2)
        pad_bottom = pad_top
        pad_left = math.floor(pad_w/2)
        pad_right = pad_left+1
        X = np.pad(X, ((int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))), 
                    'constant',constant_values = constant_value) #top bottom left right 

    elif h % 2 != 0 and w % 2 == 0:
        pad_top = math.floor(pad_h/2)
        pad_bottom = pad_top+1
        pad_left = math.floor(pad_w/2)
        pad_right = pad_left
        X = np.pad(X, ((int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))),
                    'constant',constant_values = constant_value) #top bottom left right 

    else:
        pad_top = math.floor(pad_h/2)
        pad_bottom = pad_top
        pad_left = math.floor(pad_w/2)
        pad_right = pad_left
        X = np.pad(X, ((int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))), 
                    'constant',constant_values = constant_value) #top bottom left right 

    return X

def find_best_patch(original_array):
    patch_size=512
    stride=1
    original_height, original_width = original_array.shape
    best_ratio = 0
    best_patch = None

    # Check if original array is smaller than patch_size in both dimensions
    if original_height < patch_size and original_width < patch_size:
        best_patch = padding(original_array)

    # Check which dimension needs stride-based computation
    elif original_height < patch_size:
        best_patch, best_ratio = compute_stride(original_array, patch_size, stride, dimension='width')

    elif original_width < patch_size:
        best_patch, best_ratio = compute_stride(original_array, patch_size, stride, dimension='height')

    else:
        best_patch, best_ratio = compute_stride(original_array, patch_size, stride, dimension='both')

    return best_patch, best_ratio

def compute_stride(original_array, patch_size, stride, dimension):
    original_height, original_width = original_array.shape
    best_ratio = 0
    best_patch = None

    if dimension == 'width':
        loop_range = range(0, original_width - patch_size + 1, stride)
        for x in loop_range:
            cropped_patch = crop_and_pad_patch(original_array, patch_size, x)
            best_patch, best_ratio = update_best_patch(original_array, cropped_patch, best_patch, best_ratio, x=x)

    elif dimension == 'height':
        loop_range = range(0, original_height - patch_size + 1, stride)
        for y in loop_range:
            cropped_patch = crop_and_pad_patch(original_array, patch_size, y=y)
            best_patch, best_ratio = update_best_patch(original_array, cropped_patch, best_patch, best_ratio, y=y)

    elif dimension == 'both':
        for y in range(0, original_height - patch_size + 1, stride):
            for x in range(0, original_width - patch_size + 1, stride):
                cropped_patch = crop_and_pad_patch(original_array, patch_size, x, y)
                best_patch, best_ratio = update_best_patch(original_array, cropped_patch, best_patch, best_ratio, x=x, y=y)

    return best_patch, best_ratio

def update_best_patch(original_array, cropped_patch, best_patch, best_ratio, x=None, y=None):
    patch_sum = np.nansum(np.abs(cropped_patch))
    original_sum = np.nansum(np.abs(original_array))

    ratio = patch_sum / original_sum if original_sum != 0 else 0
    if (ratio - best_ratio) > 0.0:
        best_ratio = ratio
        best_patch = cropped_patch
        if x is not None and y is not None:
            print(f"Selected patch with ratio: {best_ratio} at x={x}, y={y}")
        elif x is not None:
            print(f"Selected patch with ratio: {best_ratio} at x={x}")
        elif y is not None:
            print(f"Selected patch with ratio: {best_ratio} at y={y}")

    return best_patch, best_ratio

def crop_and_pad_patch(original_array, patch_size, x=None, y=None):
    if x is not None:
        x_start = max(0, x)
        x_end = min(original_array.shape[1], x + patch_size)
    else:
        x_start = 0
        x_end = original_array.shape[1]

    if y is not None:
        y_start = max(0, y)
        y_end = min(original_array.shape[0], y + patch_size)
    else:
        y_start = 0
        y_end = original_array.shape[0]

    cropped_patch = original_array[y_start:y_end, x_start:x_end]

    pad_height = patch_size - cropped_patch.shape[0]
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    pad_width = patch_size - cropped_patch.shape[1]
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    cropped_patch = np.pad(cropped_patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return cropped_patch


# Function to save image as JPG
def save_img(img, filename):
    im = Image.fromarray(img).convert('L')
    im.save(filename)

# Function to process a single row from the CSV
def process_row(row):
    try:
        HARP = str(row['HARP'])
        sharp_file = str(row['SHARP_FILE'])
        bitmap_file = str(row['Bitmap'])

        # Read AR Patch
        with fits.open(os.path.join(source_base_directory, HARP, sharp_file), cache=False) as AR_patch_fits:
            AR_patch_fits.verify('fix')
            AR_patch_data = AR_patch_fits[1].data
            AR_patch_data = np.flipud(AR_patch_data)
            AR_patch_data_header = AR_patch_fits[1].header  # Access header data

        # Read Bitmap
        with fits.open(os.path.join(source_base_directory, HARP, bitmap_file), cache=False) as bitmap_hdul:
            bitmap_hdul.verify('fix')
            bitmap_data = bitmap_hdul[0].data
            bitmap_data = np.flipud(bitmap_data)

        # Process for the first pair of lb and ub
        lb, ub = (-25, 25)

        # Save height and width to a dictionary
        # Save height and width to a dictionary
        metadata = {
            'SHARP_FILE': sharp_file,
            'LATDTMIN': AR_patch_data_header['LATDTMIN'],
            'LONDTMIN': AR_patch_data_header['LONDTMIN'],
            'LATDTMAX': AR_patch_data_header['LATDTMAX'],
            'LONDTMAX': AR_patch_data_header['LONDTMAX'],
            'LAT_MIN': AR_patch_data_header['LAT_MIN'],
            'LON_MIN': AR_patch_data_header['LON_MIN'],
            'LAT_MAX': AR_patch_data_header['LAT_MAX'],
            'LON_MAX': AR_patch_data_header['LON_MAX'],
            'LAT_FWT': AR_patch_data_header['LAT_FWT'],
            'LON_FWT': AR_patch_data_header['LON_FWT'],
            'NOAA_AR': AR_patch_data_header['NOAA_AR'],
            'NOAA_ARS': AR_patch_data_header['NOAA_ARS'],
            'NOAA_NUM': AR_patch_data_header['NOAA_NUM'],
            'Height': AR_patch_data.shape[0],
            'Width': AR_patch_data.shape[1],
            'Cropped_Height': None,  # Placeholder for cropped height
            'Cropped_Width': None,  # Placeholder for cropped width
            'USFLUX_RATIO':None #Placeholder for preserved unsigned flux %
        }

        # Process the second product: Bitmap Filtered
        cropped = bitmap_cropping(AR_patch_data, bitmap_data)
        cropped_fixed = np.nan_to_num(cropped)
        hmi_data_preprocessed = preprocess(cropped_fixed,lb, ub)
        best_patch, ratio = find_best_patch(hmi_data_preprocessed)
        hmi_data_scaled = bytscal_with_nan(best_patch)


        # Create destination directory structure for the second product
        output_dir_cropped = os.path.join(destination_base_directory, f'stride_based_hourly_all/{HARP}')
        os.makedirs(output_dir_cropped, exist_ok=True)

        # Save the cropped image
        cropped_image_filename = os.path.join(output_dir_cropped, sharp_file.replace('.fits', '.jpg'))
        save_img(hmi_data_scaled, cropped_image_filename)
        print(f"Saving: {cropped_image_filename}") 

        # Update the height and width for the cropped image
        metadata['Cropped_Height'] = cropped.shape[0]
        metadata['Cropped_Width'] = cropped.shape[1]
        metadata['Final_Height'] = hmi_data_scaled.shape[0]
        metadata['Final_Width'] = hmi_data_scaled.shape[1]
        metadata['USFLUX_RATIO'] = ratio
        # Return the metadata
        return metadata
    except Exception as e:
        # Handle any exceptions and log them
        print(f"Error processing row: {e}")
        return None

if __name__ == '__main__':
    # Load the CSV file
    csv_file = 'filtered_hourly_for_jpg_compression.csv'
    df = pd.read_csv(csv_file)

    # Base directories
    source_base_directory = '/data/SHARPS/raw-sharps/'
    destination_base_directory = '/data/SHARPS/preprocessed_SHARPS_JPGS/'

    # Create a pool of worker processes
    num_processes = 80  # You can adjust this to your preferred number of processes
    with Pool(processes=num_processes) as pool:
        metadata_list = pool.map(process_row, df.to_dict(orient='records'))

    # Filter out None values from metadata_list
    metadata_list = [metadata for metadata in metadata_list if metadata is not None]

    # Create a DataFrame from the filtered metadata list
    metadata_df = pd.DataFrame(metadata_list)

    # Save the metadata to a CSV file
    metadata_df.to_csv('meta_data_updated.csv', index=False)
