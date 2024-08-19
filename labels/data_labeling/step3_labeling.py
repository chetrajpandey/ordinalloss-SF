import pandas as pd

# Read CSV files
events_df = pd.read_csv('intermediates/harp2noaaAR_mappedGOES.csv')
instances_df = pd.read_csv('intermediates/stride/filtered_hourly_instances.csv', low_memory=False)

# Convert columns to datetime if they're not already in datetime format
events_df['start_time'] = pd.to_datetime(events_df['start_time'])
instances_df['timestamp'] = pd.to_datetime(instances_df['timestamp'])

# Prepare an empty list to store the resulting goes_class values
goes_classes = []

# Iterate through each instance in instances_df
for index, instance in instances_df.iterrows():
    harp_number = instance['harpnumber']
    timestamp = instance['timestamp']
    
    # Filter events within 24 hours of the instance timestamp
    filtered_events = events_df[
        (events_df['DEF_HARPNUM'] == harp_number) &
        (events_df['start_time'] >= timestamp - pd.Timedelta('1 days')) &
        (events_df['start_time'] <= timestamp)
    ]
    
    if len(filtered_events) > 0:
        # If events are found, get the maximum goes_class
        max_goes_class = filtered_events['goes_class'].max()
        goes_classes.append(str(max_goes_class))  # Append the max goes_class as a string
    else:
        goes_classes.append('')  # No events found, assign an empty string

# Add the goes_classes list as a new column in instances_df
instances_df['goes_class'] = goes_classes

# Save the instances_df with the goes_class assigned to a new CSV file
instances_df.to_csv('intermediates/stride/instances_with_goes_class.csv', index=False)
