In this code, I used different motion prediction algorithms such as Multi Layer Perceptron (MLP), Constant Velocity Model, Constant Acceleration Model, Long Short Term Memory (LSTM) and Kalman Filter for predicting the motion of Autonomous Vehicles. I have predicted the motion for two different time stamps: 13 frames and 125 frames.

** We have done the denormalization in the file Denormalization.py **

Clone the repo using git and follow the process below:

Steps to run the models:

1. We have 5 models (Constant Velocity, Constant Acceleration, MLP, LSTM and Kalman Filter) and everything are defined inside nn_modules.py. The models are called in the main.py. 

2. We have done the data preprocessing techniques like data normalization and label encoding in the file Normalization_&_Label_Encoding_.py. 

3. We used separate Constant_VelocityModel() for predicting the motion of the pedestrians and bicyclists in the nn_modlues.py which is called in the main.py. 

4. Our new approach is Kalman Filter, we used this as a separate model for the prediction of the motion of the cars and trucks_buses, which is defined in nn_modules.py and called in the main.py
