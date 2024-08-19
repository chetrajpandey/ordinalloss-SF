import os
import pandas as pd
# import numpy as np

def create_CVDataset_for_lon_range(min_lon, max_lon):
    # Construct the file path based on the provided lon range
    file_path = f'intermediates/stride/complete_hourly_dataset_lon_{min_lon}_to_{max_lon}.csv'

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path, low_memory=False)

    # Convert 'timestamp' to a datetime object for date manipulation
    data['harp_start'] = pd.to_datetime(data['harp_start'])
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Sort the data by timestamp
    data = data.sort_values(by='timestamp')

    # Create folder if it doesn't exist
    output_folder = f'ar_labels_M/version_6_stride/{min_lon}_to_{max_lon}'
    os.makedirs(output_folder, exist_ok=True)

    cols = ['harpnumber', 'timestamp', 'SHARP_FILE', 'goes_class', 'label', 'harp_start']
    search_list = [['01', '02', '03'], ['04', '05', '06'], ['07', '08', '09'], ['10', '11', '12']]

    for fold in range(4):
        # Determine test and train partitions based on fold
        test_partition = fold
        train_partitions = [i for i in range(4) if i != test_partition]

        # Mask for test set
        test_mask = data['harp_start'].dt.strftime('%m').isin(search_list[test_partition])
        test_set = data[test_mask]

        # Mask for train set
        train_mask = data['harp_start'].dt.strftime('%m').isin([month for i in train_partitions for month in search_list[i]])
        train_set = data[train_mask]

        print(f'Fold {fold+1} Dataset:')
        print('Test Set: \n', test_set['label'].value_counts())
        # print('Train Set: \n', train_set['label'].value_counts())
        print('\n\n')

        # Output train and test sets into CSV inside the created folder
        train_set.to_csv(f'{output_folder}/Fold{fold+1}_train.csv', index=False, header=True, columns=cols)
        test_set.to_csv(f'{output_folder}/Fold{fold+1}_test.csv', index=False, header=True, columns=cols)

# Define a list of lon range pairs
lon_ranges = [(-30, 30), (-45, 45), (-60, 60), (-75, 75), (-90, 90)]  # Add more pairs as needed

# Iterate through lon range pairs and call the function
for min_lon, max_lon in lon_ranges:
    create_CVDataset_for_lon_range(min_lon, max_lon)
