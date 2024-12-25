Group 23: --> Mohana Prasath Saravanan, Saniket Sunil Agarkar

The process to run the models are same as in the README.md files which you shared with us for this Final project  'CODE'. 

** We have done the denormalization in the file Denormalization.py **

Steps to run the models:

1. We have 5 models (Constant Velocity, Constant Acceleration, MLP, LSTM and Kalman Filter) and everything are defined inside nn_modules.py. The models are called in the main.py. 

2. We have done the data preprocessing techniques data normalization and label encoding in the file Normalization_&_Label_Encoding_.py. 

3. We used separate Constant_VelocityModel() for predicting the motion of the pedestrians and bicyclists in the nn_modlues.py which is called in the main.py. 

4. Our new approach is Kalman Filter, we used this as a separate model for the prediction of the motion of the cars and trucks_buses, which is defined in nn_modules.py and called in the main.py.  
