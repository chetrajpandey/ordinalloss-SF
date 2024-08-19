import pandas as pd
import numpy as np


def process_csv(input_file='intermediates/stride/instances_with_goes_class.csv', center_range=30):
    # Read the CSV file
    df = pd.read_csv(input_file, low_memory=False)
    # print(df['LON_MIN'].max(), df['LON_MIN'].min())

    # Replace '.fits' with '.jpg' in the 'SHARP_FILE' column
    df['SHARP_FILE'] = df['SHARP_FILE'].str.replace('.fits', '.jpg', regex=True)

    # Explicitly convert 'goes_class' to string and replace NaN with an empty string
    df['goes_class'] = df['goes_class'].astype(str).replace('nan', '').replace('NaN', '')

    # Create a new column 'label' based on 'goes_class'
    df['label'] = np.where((df['goes_class'] >= 'M1.0') & (df['goes_class'] != ''), 1, 0)

    # First filter for width and height less than or equal to 1024
    filtered_df = df[(df['Cropped_Width'] >= 70)]
    # filtered_df = filtered_df[(filtered_df['USFLUX_RATIO']>0.7) | (filtered_df['USFLUX_RATIO']==0.0)]
    # filtered_df = df.copy()

    # Second filter for label is 0 and width/height greater than 64, or label is 1
    # final_filtered_df = filtered_df[((filtered_df['label'] == 0) & (filtered_df['Cropped_Width'] >= 64)) | (filtered_df['label'] == 1)]

    # Additional filter based on 'LON_MIN'
    filtered_df = filtered_df[((filtered_df['LON_FWT'] >= -center_range) & (filtered_df['LON_FWT'] <= center_range))
                              ]
    # filtered_df = filtered_df[(filtered_df['LON_MIN'] >= lon_min) & (filtered_df['LON_MIN'] <= lon_max) & (filtered_df['LON_MAX'] >= lon_min) & (filtered_df['LON_MAX'] <= lon_max)]
    print("For lon within: ", center_range)
    print(df['label'].value_counts())
    print(filtered_df['label'].value_counts())
    print('\n')

    # Create a dynamic output file name based on filter values
    output_file = f'intermediates/stride/complete_hourly_dataset_lon_-{center_range}_to_{center_range}.csv'

    # print(final_filtered_df['label'].value_counts())
    # Save the filtered DataFrame to a new CSV file
    # filtered_df.to_csv(output_file, index=False, header=True, columns=['harpnumber', 'timestamp', 'SHARP_FILE', 'goes_class', 'label', 'harp_start', 'LON_MIN', 'LON_MAX', 'LON_FWT', 'USFLUX_RATIO'])

# Call the function with default file names and lon_min filter values
process_csv(center_range=30)
# process_csv(center_range=45)
process_csv(center_range=60)
# process_csv(center_range=70)
# process_csv(center_range=75)
process_csv(center_range=90)