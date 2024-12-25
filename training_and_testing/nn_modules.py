import math
import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from utils import build_module
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# TODO: Here you should create your models. You can use the MLPModel or ConstantVelocity as a template.
#  Each model should have a __init__ function, a forward function, and a loss_function function.
#  The loss function doen't have to be in the model, but it is convenient to have it there, because the lit_module
#  will call it automatically, because you assign a prediction model to it and later it asks the model for the loss function.


def Constant_VelocityModel():
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read data from CSV files
    recording_id = "28"
    data_track = pd.read_csv(
        '/home/kus05jof/Seafile/My Library/CODE/pythonProject/data_processing/dataset/data/' + recording_id + '_tracks.csv')
    data_meta = pd.read_csv(
        '/home/kus05jof/Seafile/My Library/CODE/pythonProject/data_processing/dataset/data/' + recording_id + '_tracksMeta.csv')

    # Filter for 'car' and 'truck_bus'
    desired_classes = ['pedestrian', 'bicycle']
    new_data_track = data_meta[data_meta['class'].isin(desired_classes)]
    new_data_track_id = new_data_track['trackId']

    # Filter data_track based on the filtered track IDs
    data_track_filtered = data_track[data_track['trackId'].isin(new_data_track_id)]

    data = data_track_filtered

    # Define parameters
    dt = 0.04  # Time step (delta t)
    num_time_steps_to_predict = 13

    # Lists to store predictions and measurements
    predicted_x = []
    predicted_y = []
    measurement_x = []
    measurement_y = []

    # Initial state (from the first data point)
    x = data['xCenter'].iloc[0]
    y = data['yCenter'].iloc[0]
    vx = data['xVelocity'].iloc[0]
    vy = data['yVelocity'].iloc[0]

    # Predict future positions using constant velocity model
    for index, row in data.iterrows():
        # Record the current measurement
        measurement_x.append(row['xCenter'])
        measurement_y.append(row['yCenter'])

        # Predict the next position using constant velocity model
        x += vx * dt
        y += vy * dt

        # Store the predicted position
        predicted_x.append(x)
        predicted_y.append(y)

        # Break if the desired number of predictions is reached
        if len(predicted_x) >= num_time_steps_to_predict:
            break

    # Calculate errors
    predicted_x = torch.tensor(predicted_x, dtype=torch.float32, device=device)
    predicted_y = torch.tensor(predicted_y, dtype=torch.float32, device=device)
    measurement_x = torch.tensor(measurement_x, dtype=torch.float32, device=device)
    measurement_y = torch.tensor(measurement_y, dtype=torch.float32, device=device)

    mse_x = torch.mean((measurement_x - predicted_x) ** 2)
    mse_y = torch.mean((measurement_y - predicted_y) ** 2)
    mse_total = torch.sqrt(mse_x ** 2 + mse_y ** 2)

    print(f"Total Mean Squared Error (MSE): {mse_total.item()} meters")

    displacement_errors = torch.sqrt((predicted_x - measurement_x) ** 2 + (predicted_y - measurement_y) ** 2)
    average_displacement_error = torch.mean(displacement_errors)
    print(f"Average Displacement Error: {average_displacement_error.item()} meters")

    # Plot the results
    time_range = torch.linspace(1 / 25, num_time_steps_to_predict / 25, steps=num_time_steps_to_predict, dtype=torch.float32)
    plt.figure(figsize=(10, 5))
    plt.plot(time_range.cpu(), displacement_errors.cpu(), label='Displacement Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement Error (meters)')
    plt.title('Displacement Error vs. Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('ConstantVelocity_DisplacementError.png', dpi=300)
    plt.show()

    # Save the results to a DataFrame
    result_df = pd.DataFrame({
        'measurement_x': measurement_x.cpu().numpy(),
        'predicted_x': predicted_x.cpu().numpy(),
        'measurement_y': measurement_y.cpu().numpy(),
        'predicted_y': predicted_y.cpu().numpy()
    })
    # Plot the trajectories
    fig = plt.figure(figsize=(8, 6))

    # Ground Truth Path
    plt.plot(measurement_x.cpu(), measurement_y.cpu(), label='Bicycle Path_Ground Truth', color='blue')
    plt.scatter(measurement_x[0].cpu(), measurement_y[0].cpu(), color='blue', label='Start_Ground Truth')
    plt.scatter(measurement_x[-1].cpu(), measurement_y[-1].cpu(), color='orange', label='End_Ground Truth')

    # Predicted Path
    plt.plot(predicted_x.cpu(), predicted_y.cpu(), label='Bicycle Path_Predicted', color='orange')
    plt.scatter(predicted_x[0].cpu(), predicted_y[0].cpu(), color='green', label='Start_Predicted')
    plt.scatter(predicted_x[-1].cpu(), predicted_y[-1].cpu(), color='red', label='End_Predicted')

    # Set plot details
    plt.title("Bicycle Trajectory")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Save the figure
    plt.savefig('Bicycle_Trajectory_Comparison.png', dpi=300)
    plt.show()

    # Save the results to a DataFrame
    result_df = pd.DataFrame({
        'measurement_x': measurement_x.cpu().numpy(),
        'predicted_x': predicted_x.cpu().numpy(),
        'measurement_y': measurement_y.cpu().numpy(),
        'predicted_y': predicted_y.cpu().numpy()
    })
    result_df.to_csv('trajectory_comparison.csv', index=False)

    print("Trajectory comparison and data saved successfully.")

    # Save the image
    output_image_file = os.path.join(output_folder, 'ConstantVelocity_DisplacementError.png')
    plt.savefig(output_image_file)
    print(f"Image saved to {output_image_file}")
    # Define the folder path
    output_folder = 'CV results'

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the Excel file
    output_excel_file = os.path.join(output_folder, 'CV_output_data.xlsx')
    result_df.to_excel(output_excel_file, index=False)
    print(f"Data saved to {output_excel_file}")

    # Save the image
    output_image_file = os.path.join(output_folder, 'ConstantVelocity_DisplacementError.png')
    plt.savefig(output_image_file)
    print(f"Image saved to {output_image_file}")


class ConstantVelocityModel(nn.Module):
    def __init__(self, dt=0.12):
        super(ConstantVelocityModel, self).__init__()
        self.dt = dt

    def forward(self, x):
        x = x[:, -1, :]
        x_plus = x + self.dt * x
        return x_plus


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        x = x.view(batch_size, -1, self.output_dim)
        return x

class ConstantAccelerationModel(nn.Module):
    def __init__(self, dt=0.12):
        super(ConstantAccelerationModel, self).__init__()
        self.dt = dt

    def forward(self,x):
        x=x[:,-1,:]
        x_plus = x + self.dt * x + 0.5 * self.dt **2 * x
        return x_plus


class LSTM(nn.Module):
    def __init__(self, input_Size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_Size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_Size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)



    def forward(self, x):
        # batch_size = x.size(0)
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = out.view(out.size(0), -1, self.output_size)
        return out

def KalmanFilter():
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recording_id="19"
    # Load the files
    file_path_1 = '/home/kus05jof/Seafile/My Library/CODE/pythonProject/data_processing/dataset/data/' + recording_id + '_tracks.csv'
    file_path_2 = '/home/kus05jof/Seafile/My Library/CODE/pythonProject/data_processing/dataset/data/' + recording_id + '_tracksMeta.csv'
    features_1 = ['trackId', 'class']

    df_00 = pd.read_csv(file_path_2, usecols=features_1)
    df_02 = pd.read_csv(file_path_1)

    # Merge the dataframes
    merged_df = pd.merge(df_00, df_02, on='trackId', how='left')

    # Step 2: Filter rows where 'class' is 'pedestrian' or 'bicycle'
    filtered_df = merged_df[merged_df['class'].isin(['car'])]

    # Step 3: Perform Label Encoding on the 'class' column
    label_encoder = LabelEncoder()
    filtered_df['class'] = label_encoder.fit_transform(filtered_df['class'])

    # Step 4: Normalize only selected columns
    scaler = MinMaxScaler()

    # List of columns to normalize (case-sensitive)
    features_2 = ['xVelocity', 'yVelocity', 'yCenter', 'xCenter', 'heading', 'xAcceleration', 'yAcceleration',
                  'lonVelocity', 'latVelocity']

    # Create a copy of filtered_df to store the normalized data
    normalized_df = filtered_df.copy()

    # Apply normalization only on the selected features
    normalized_df[features_2] = scaler.fit_transform(filtered_df[features_2])

    data = normalized_df

    # Define Kalman filter parameters
    x_initial = torch.tensor(data['xCenter'].iloc[0], dtype=torch.float32, device=device)
    y_initial = torch.tensor(data['yCenter'].iloc[0], dtype=torch.float32, device=device)
    x_velocity_initial = torch.tensor(data['xVelocity'].iloc[0], dtype=torch.float32, device=device)
    y_velocity_initial = torch.tensor(data['yVelocity'].iloc[0], dtype=torch.float32, device=device)

    dt = 0.04  # Time step (delta t)

    initial_state = torch.tensor([x_initial, y_initial, x_velocity_initial, y_velocity_initial], dtype=torch.float32,
                                 device=device)
    initial_covariance = torch.eye(4, dtype=torch.float32, device=device)  # Initial state covariance matrix

    process_noise = torch.tensor([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 0.1, 0],
                                  [0, 0, 0, 0.1]], dtype=torch.float32, device=device)

    measurement_noise = torch.eye(4, dtype=torch.float32, device=device) * 0.01

    A = torch.tensor([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=torch.float32, device=device)

    H = torch.eye(4, dtype=torch.float32, device=device)

    state = initial_state
    covariance = initial_covariance

    predicted_x = []
    predicted_y = []
    measurement_x = []
    measurement_y = []
    deviated_distances = []

    frames_predicted = 0
    num_time_steps_to_predict = 125

    for index, row in data.iterrows():
        measurement = torch.tensor([row['xCenter'], row['yCenter'], row['xVelocity'], row['yVelocity']],
                                   dtype=torch.float32, device=device)
        measurement_x.append(measurement[0].item())
        measurement_y.append(measurement[1].item())

        state_estimate = torch.mm(A, state.unsqueeze(1)).squeeze(1)
        covariance_estimate = torch.mm(A, torch.mm(covariance, A.T)) + process_noise

        kalman_gain = torch.mm(covariance_estimate, torch.mm(H.T, torch.inverse(
            torch.mm(H, torch.mm(covariance_estimate, H.T)) + measurement_noise)))
        state = state_estimate + torch.mm(kalman_gain,
                                          (measurement - torch.mm(H, state_estimate.unsqueeze(1)).squeeze(1)).unsqueeze(
                                              1)).squeeze(1)
        covariance = torch.mm((torch.eye(4, dtype=torch.float32, device=device) - torch.mm(kalman_gain, H)),
                              covariance_estimate)

        deviated_distance = torch.sqrt((measurement[0] - state[0]) ** 2 + (measurement[1] - state[1]) ** 2)
        deviated_distances.append(deviated_distance.item())

        frames_predicted += 1
        predicted_x.append(state[0].item())
        predicted_y.append(state[1].item())

        if frames_predicted >= num_time_steps_to_predict:
            break

    predicted_x = torch.tensor(predicted_x, dtype=torch.float32)
    predicted_y = torch.tensor(predicted_y, dtype=torch.float32)
    measurement_x = torch.tensor(measurement_x, dtype=torch.float32)
    measurement_y = torch.tensor(measurement_y, dtype=torch.float32)

    mse_x = torch.mean((measurement_x - predicted_x) ** 2)
    mse_y = torch.mean((measurement_y - predicted_y) ** 2)

    mse_total = torch.sqrt(mse_x ** 2 + mse_y ** 2)
    print(f"Total Mean Squared Error (MSE): {mse_total.item()} meters")

    last_deviated_distance = deviated_distances[-1]
    print(f"Deviated Distance at Last Point of Prediction: {last_deviated_distance} meters")

    displacement_errors = torch.sqrt((predicted_x - measurement_x) ** 2 + (predicted_y - measurement_y) ** 2)
    average_displacement_error = torch.mean(displacement_errors)
    print(f"Average Displacement Error: {average_displacement_error.item()} meters")

    time_range = torch.linspace(1 / 25, num_time_steps_to_predict / 25, steps=num_time_steps_to_predict,
                                dtype=torch.float32)
    plt.figure(figsize=(10, 5))
    plt.plot(time_range.cpu(), deviated_distances, label='Deviated Distance')
    plt.xlabel('Time (s)')
    plt.ylabel('Deviated Distance')
    plt.title('Deviated Distance vs. Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('Kalman Distance Deviation.png', dpi=300)
    plt.show()

    xCenter = data['xCenter'].tolist()[:num_time_steps_to_predict]
    yCenter = data['yCenter'].tolist()[:num_time_steps_to_predict]

    result_df = pd.DataFrame({
        'xCenter': xCenter,
        'measurement_x': measurement_x.cpu().numpy(),
        'predicted_x': predicted_x.cpu().numpy(),
        'yCenter': yCenter,
        'measurement_y': measurement_y.cpu().numpy(),
        'predicted_y': predicted_y.cpu().numpy()
    })


    # Folder path defined here
    output_folder = 'KL results'

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Track the file numbering (incrementing the counter each time when saved)
    file_counter = 1

    # Excel file name created using the counter
    output_excel_file = os.path.join(output_folder, f'{file_counter}.xlsx')

    # Saving the file
    result_df.to_excel(output_excel_file, index=False)
    print(f"Data saved to {output_excel_file}")

    # Create the image file name using the counter (assuming you're using matplotlib or similar)
    output_image_file = os.path.join(output_folder, f'{file_counter}.png')

    # Image saved
    plt.savefig(output_image_file)
    print(f"Image saved to {output_image_file}")

    # Increment the file_counter for the next save
    file_counter += 1



















