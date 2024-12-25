import numpy as np
import os
import math
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from nn_modules import LSTM, MultiLayerPerceptron, ConstantVelocityModel, ConstantAccelerationModel, KalmanFilter
import matplotlib
import matplotlib.pyplot as plt

# Select the models that you want to run
model = "lstm" # "lstm", "mlp", "const_vel", "const_acc", "kalman"

# Load the files
file_path_1 = '/home/kus05jof/Seafile/My Library/CODE/pythonProject/data_processing/dataset/data/20_tracksMeta.csv'
file_path_2 = '/home/kus05jof/Seafile/My Library/CODE/pythonProject/data_processing/dataset/data/20_tracks.csv'
features_1 = ['trackId', 'class']


df_00 = pd.read_csv(file_path_1, usecols=features_1)
df_02 = pd.read_csv(file_path_2)

# Merge the dataframes
# merged_df = pd.merge(df_00, df_02, on='trackId', how='left')
merged_df = df_02

# Step 2: Filter rows where 'class' is 'pedestrian', 'bicycle', 'car', or 'truck_bus'
# filtered_df = merged_df[merged_df['class'].isin(['car', 'bicycle', 'pedestrian', 'truck_bus'])]

# Step 3: Perform Label Encoding on the 'class' column
# label_encoder = LabelEncoder()
# filtered_df['class'] = label_encoder.fit_transform(filtered_df['class'])

# Step 4: Normalize only selected columns
scaler = MinMaxScaler()

# List of columns to normalize (case-sensitive)
features_2 = ['xVelocity', 'yVelocity', 'yCenter', 'xCenter', 'heading', 'xAcceleration', 'yAcceleration', 'lonVelocity', 'latVelocity']
features_nn = ['yCenter', 'xCenter', 'heading']
# Apply normalization only on the selected features
filtered_df_nn_denorm = merged_df[features_nn]
filtered_df_nn_norm = scaler.fit_transform(filtered_df_nn_denorm)

# define the model
past_sequence_length = 6
future_sequence_length = 125
num_features = 3

if model == "lstm":
    input_size = num_features
    hidden_size = 32
    output_size = num_features

    model_ = LSTM(input_size, hidden_size, output_size)
elif model == "mlp":
    input_size = num_features * past_sequence_length
    hidden_size = 32
    output_size = num_features

    model_ = MultiLayerPerceptron(input_size, hidden_size, output_size)
elif model == "const_vel":
    model_ = ConstantVelocityModel()
elif model == "const_acc":
    model_ = ConstantAccelerationModel()
elif model == "kalman":
    model_ = KalmanFilter()

y_hat_list = []

filtered_df_nn_norm_pt = torch.tensor(filtered_df_nn_norm, dtype=torch.float32).unsqueeze(dim=0)

with torch.no_grad():
    x = filtered_df_nn_norm_pt[:, :past_sequence_length, :]

    for k in range(future_sequence_length):
        y_hat_k = model_(x)
        y_hat_list.append(y_hat_k)
        if y_hat_k.dim() < 3:
            y_hat_k = y_hat_k.unsqueeze(1)
        x = torch.cat([x[:, 1:, :], y_hat_k], dim=1)

    y_hat = torch.stack(y_hat_list, dim=1).squeeze()

    y_hat_denorm = scaler.inverse_transform(y_hat)
    y_hat_gt_denorm = scaler.inverse_transform(filtered_df_nn_norm)[past_sequence_length: past_sequence_length + future_sequence_length, :]

    avg_displacement_error = torch.linalg.norm(torch.tensor(y_hat_denorm[:, :2] - y_hat_gt_denorm[:, :2]))
    print("avg displacement error: ", avg_displacement_error)

    avg_heading_error = torch.linalg.norm(torch.tensor(y_hat_denorm[:, -1] - y_hat_gt_denorm[:, -1]))
    print("avg heading error: ", avg_heading_error)

    plt.figure()
    plt.plot(y_hat_gt_denorm[:, 0])
    plt.plot(y_hat_denorm[:, 0])
    plt.legend(["x-center (ground truth)", "x-center {}".format(model)])
    plt.xlabel("frame idx")
    plt.ylabel("x-center")
    plt.show()

    plt.figure()
    plt.plot(y_hat_gt_denorm[:, 1])
    plt.plot(y_hat_denorm[:, 1])
    plt.legend(["y-center (ground truth)", "y-center {}".format(model)])
    plt.xlabel("frame idx")
    plt.ylabel("y-center")
    plt.show()

    plt.figure()
    plt.plot(y_hat_gt_denorm[:, 2])
    plt.plot(y_hat_denorm[:, 2])
    plt.legend(["heading (ground truth)", "heading {}".format(model)])
    plt.xlabel("frame idx")
    plt.ylabel("heading")
    plt.show()