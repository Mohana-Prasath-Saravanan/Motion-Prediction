import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Directory paths for input and output files
input_dir = 'C:\\Users\\saniket\\OneDrive\\Desktop\\CVT Course Material\\Motion Prediction & Control Seminar\\Bicycle_Model\\Dataset\\data\\'
output_dir = 'C:\\Users\\saniket\\OneDrive\\Desktop\\CVT Course Material\\Motion Prediction & Control Seminar\\Bicycle_Model\\Modified .csv files\\'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Columns to load from the metadata files (xx_tracksMeta.csv)
features_1 = ['trackId', 'class']

# Columns to normalize in the merged data
features_2 = ['xVelocity', 'yVelocity', 'yCenter', 'xCenter', 'heading', 'xAcceleration', 'yAcceleration', 'lonVelocity', 'latVelocity']

# Initialize LabelEncoder and MinMaxScaler
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

# Loop over the file index range from 01 to 32
for i in range(1, 33):
    # Generate the correct file name (padded with 0 for single digits)
    file_index = f"{i:02}"

    # Define file paths for the metadata and track files
    file_path_meta = os.path.join(input_dir, f"{file_index}_tracksMeta.csv")
    file_path_tracks = os.path.join(input_dir, f"{file_index}_tracks.csv")

    # Check if both files exist
    if os.path.exists(file_path_meta) and os.path.exists(file_path_tracks):
        # Load the metadata and tracks data
        df_meta = pd.read_csv(file_path_meta, usecols=features_1)
        df_tracks = pd.read_csv(file_path_tracks)

        # Merge the dataframes on 'trackId'
        merged_df = pd.merge(df_meta, df_tracks, on='trackId', how='left')

        # Filter rows where 'class' is 'pedestrian', 'bicycle', 'car', or 'truck_bus'
        filtered_df = merged_df[merged_df['class'].isin(['car', 'bicycle', 'pedestrian', 'truck_bus'])]

        # Perform label encoding on the 'class' column
        filtered_df['class'] = label_encoder.fit_transform(filtered_df['class'])

        # Normalize selected columns
        filtered_df[features_2] = scaler.fit_transform(filtered_df[features_2])

        # Specify the output file name and path
        output_file_path = os.path.join(output_dir, f"{file_index}_tracks.csv")

        # Save the normalized DataFrame
        filtered_df.to_csv(output_file_path, index=False)

        # Print confirmation for each file processed
        print(f"Processed and saved file: {output_file_path}")

    else:
        print(f"Files for index {file_index} not found, skipping...")
