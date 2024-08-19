
import pandas as pd

# Read the CSV file
data = pd.read_csv('../../fits_ar_processor/meta_data_updated.csv', low_memory=False)

# Extracting the 'harpnumber' using regex
data['harpnumber'] = data['SHARP_FILE'].str.extract(r'hmi\.sharp_cea_720s\.(\d+)\.\d{8}_\d{6}')

# Convert the extracted 'harpnumber' to numeric type
data['harpnumber'] = pd.to_numeric(data['harpnumber'])

# Extract the timestamp from the filename
data['timestamp'] = data['SHARP_FILE'].str.extract(r'(\d{8}_\d{6})')

# Ensure the extracted timestamp is in the correct format (YYYYMMDD_HHmmss)
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')

# Find the minimum timestamp for each unique harpnumber
min_timestamps = data.groupby('harpnumber')['timestamp'].min().reset_index()

# Merge the original data with the minimum timestamps
data = pd.merge(data, min_timestamps, on='harpnumber', suffixes=('', '_min'))

# Filter for rows where the timestamp is at the start of the hour
filtered_data = data[data['timestamp'].dt.minute == 0]

# Rename the 'timestamp_min' column to 'harp_start'
filtered_data = filtered_data.rename(columns={'timestamp_min': 'harp_start'})

# Sort the data based on 'harpnumber' and 'SHARP_FILE'
filtered_data = filtered_data.sort_values(by=['harpnumber', 'SHARP_FILE'])

# Save the sorted and filtered data to a new CSV file
filtered_data.to_csv('intermediates/stride/filtered_hourly_instances.csv', index=False)
