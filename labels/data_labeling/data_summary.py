import pandas as pd
import numpy as np


def process_csv(input_file='intermediates/stride/instances_with_goes_class.csv', center_range=30):
    # Read the CSV file
    df = pd.read_csv(input_file, low_memory=False)
    # print(df['LON_MIN'].max(), df['LON_MIN'].min())

    # Replace '.fits' with '.jpg' in the 'SHARP_FILE' column
    df['SHARP_FILE'] = df['SHARP_FILE'].str.replace('.fits', '.jpg', regex=True)

    # Explicitly convert 'goes_class' to string and replace NaN with an empty string
    df['goes_class'] = df['goes_class'].astype(str).replace('nan', 'FQ').replace('NaN', 'NF')

    # Create a new column 'label' based on 'goes_class'
    df['label'] = np.where((df['goes_class'] >= 'M1.0') & (df['goes_class'] != ''), 1, 0)


    print(df['label'].value_counts())
    # First filter for width and height less than or equal to 1024
    df = df[(df['Cropped_Width'] >= 70)]
    print(df['label'].value_counts())

    selected_rows_x = df[df['goes_class'].str.startswith('X')]
    selected_rows_m = df[df['goes_class'].str.startswith('M')]

    # Select rows where goes_class is 'C', 'B', or empty string
    selected_rows_c = df[df['goes_class'].str.startswith('C')]
    selected_rows_b = df[df['goes_class'].str.startswith('B')]
    selected_rows_a = df[df['goes_class'].str.startswith('A')]
    selected_rows_nf = df[df['goes_class'].str.startswith('FQ')]

    print(len(selected_rows_x), len(selected_rows_m), len(selected_rows_c), len(selected_rows_b), len(selected_rows_a), len(selected_rows_nf))
    # df = df[(df['USFLUX_RATIO']>0.7) | (df['USFLUX_RATIO']==0.0)]
    # df = df.copy()

    # Second filter for label is 0 and width/height greater than 64, or label is 1
    # final_df = df[((df['label'] == 0) & (df['Cropped_Width'] >= 64)) | (df['label'] == 1)]

    # Additional filter based on 'LON_MIN'
    df = df[((df['LON_FWT'] >= -center_range) & (df['LON_FWT'] <= center_range))
                              ]
    # df = df[(df['LON_MIN'] >= lon_min) & (df['LON_MIN'] <= lon_max) & (df['LON_MAX'] >= lon_min) & (df['LON_MAX'] <= lon_max)]
    print("For lon within: ", center_range)
    print(df['label'].value_counts())
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