"""
Convert our pre-recorded scenarios into csv files with the recording id. The velocities are calculated and recorded in
the dataframe, just to have a histogram to visualize the presence of outlier values.
"""

import os
import pandas as pd
import math
import matplotlib.pyplot as plt

# Define the base directory (it will clean everything in the base directory and its subdirectories, saves them in the same place)
base_dir = "/home/sina/env_prediction_project/trajectory_detection/Recorded Datasets"
# base_dir = '/home/sina/env_prediction_project/trajectory_detection/Recorded Datasets/inlab_eval_3'

# If you want to have a histogram of the velocities, set this to True
vel_histogram = False

# Iterate through all subdirectories and files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.startswith("pose_") and file.endswith("_raw.csv"):
            file_path = os.path.join(root, file)
            
            # Read the file into a DataFrame, Drop rows with None values, and reset the index
            df = pd.read_csv(file_path, sep=",").dropna().reset_index(drop=True)
            
            # Initialize v_x, v_y, and v_total columns
            df["v_x"] = None
            df["v_y"] = None
            df["v_total"] = None
            
            # Calculate v_x, v_y, and v_total (JUST TO HAVE A HISTOGRAM)
            for i in range(1, len(df)):
                if df.loc[i, "x"] != "None" and df.loc[i - 1, "x"] != "None" and \
                   df.loc[i, "y"] != "None" and df.loc[i - 1, "y"] != "None":
                    try:
                        # Convert values to float for calculation
                        x1, x0 = float(df.loc[i, "x"]), float(df.loc[i - 1, "x"])
                        y1, y0 = float(df.loc[i, "y"]), float(df.loc[i - 1, "y"])
                        t1, t0 = float(df.loc[i, "timestamp"]), float(df.loc[i - 1, "timestamp"])
                        
                        # Calculate velocities
                        v_x = (x1 - x0) / (t1 - t0)
                        v_y = (y1 - y0) / (t1 - t0)
                        v_total = math.sqrt(v_x**2 + v_y**2)

                        # Assign calculated values
                        df.loc[i, "v_x"] = v_x
                        df.loc[i, "v_y"] = v_y
                        df.loc[i, "v_total"] = v_total
                    except ZeroDivisionError:
                        df.loc[i, "v_x"] = None
                        df.loc[i, "v_y"] = None
                        df.loc[i, "v_total"] = None
                else:
                    df.loc[i, "v_x"] = None
                    df.loc[i, "v_y"] = None
                    df.loc[i, "v_total"] = None

            # Save the updated DataFrame to a new CSV file (without velocities)
            new_file_path = os.path.join(root, file.replace("_raw.csv", "_clean.csv"))
            df.drop(columns=["v_total", "v_x", "v_y"]).to_csv(new_file_path, index=False)


            if vel_histogram:
                # Plot the distribution of v_total
                v_total_values = df["v_total"].dropna().astype(float)  # Drop None values and convert to float
                plt.figure(figsize=(10, 6))
                plt.hist(v_total_values, bins=30, color='blue', alpha=0.7)
                plt.title(f"Velocity Distribution for {file}")
                plt.xlabel("Velocity (v_total)")
                plt.ylabel("Frequency")
                
                # Save the histogram as an image
                histogram_path = os.path.join(root, file.replace("_raw.csv", "_vel_distribution.png"))
                plt.savefig(histogram_path)
                plt.close()