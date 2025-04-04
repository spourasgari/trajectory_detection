"""
Concatenate CSV files with specific scenario_id values into a single CSV file,
dropping rows with None values and keeping all columns.
"""

import os
import pandas as pd

# Define the base directory
base_dir = "/home/sina/env_prediction_project/trajectory_detection/Recorded Datasets"

# Define the specific scenario_id values to filter
target_scenario_ids = {"inlab_eval_1", "inlab_eval_2", "inlab_eval_3"}  # Add more scenario_ids as needed

# Set the sampling rate
# sr = '2.5hz'
# sr = '10hz'
sr = 'full'

output_file_name = 'straight_line_dataset.csv'

# Initialize a list to hold the data from each CSV file
filtered_data = []

# Iterate through all subdirectories and files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(sr+".csv"):
            file_path = os.path.join(root, file)
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Check if the scenario_id column contains any of the target values
            if df["scenario_id"].isin(target_scenario_ids).any():
                # Drop rows with None values
                df_cleaned = df.dropna()
                
                # Append the cleaned DataFrame to the list
                filtered_data.append(df_cleaned)

# Concatenate all filtered DataFrames into a single DataFrame
result_df = pd.concat(filtered_data, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
output_path = os.path.join(base_dir, output_file_name)
result_df.to_csv(output_path, index=False)

print(f"Filtered data saved to {output_path}")