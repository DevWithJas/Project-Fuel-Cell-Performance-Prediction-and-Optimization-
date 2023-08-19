#!/usr/bin/env python
# coding: utf-8

# # 1. Parameter Definition and Data Generation
# 

# In[1]:


import numpy as np

# Define the parameter ranges
param_ranges = {
    'Cell Voltage (V)': np.linspace(0.6, 0.9, 1000),
    'Current Density (A/cm²)': np.linspace(0.2, 2, 1000),
    'Power Density (W/cm²)': np.linspace(0.2, 2, 1000),
    'Operating Temperature (°C)': np.linspace(60, 80, 1000),
    'Hydrogen Flow Rate (mol/s)': np.linspace(0.02, 0.05, 1000),
    'Oxygen Flow Rate (mol/s)': np.linspace(0.1, 0.2, 1000),
    'Humidity of Feed Gases (%)': np.linspace(80, 100, 1000),
    'Membrane Thickness (μm)': np.linspace(25, 200, 1000),
    'Catalyst Loading (mg/cm²)': np.linspace(0.1, 0.5, 1000),
    'Membrane Swelling Ratio (%)': np.linspace(10, 50, 1000),
    'Anode Pt Loading (mg/cm²)': np.linspace(0.02, 0.06, 1000),
    'Cathode Pt Loading (mg/cm²)': np.linspace(0.04, 0.1, 1000),
    'Membrane Conductivity (S/cm)': np.linspace(0.05, 0.2, 1000),
    'Anode/Cathode Gas Diffusion Layer Thickness (μm)': np.linspace(100, 300, 1000),
    'Anode/Cathode Porosity (%)': np.linspace(30, 70, 1000),
    'Membrane Equivalent Weight (g/mol)': np.linspace(900, 1100, 1000),
    'Anode/Cathode Loading Ratio': np.linspace(1, 2, 1000),
    'Cell Active Area (cm²)': np.linspace(25, 100, 1000),
    'Cell Pressure (atm)': np.linspace(1.5, 3.0, 1000),
    'Gas Humidification Temperature (°C)': np.linspace(40, 50, 1000),
}

# Generate synthetic data
num_samples = 1000
synthetic_data = {param: np.random.choice(values, size=num_samples) for param, values in param_ranges.items()}

# Simulate power output using a simplified equation
power_output = (
    0.5 * synthetic_data['Current Density (A/cm²)'] * synthetic_data['Cell Voltage (V)'] +
    0.2 * synthetic_data['Power Density (W/cm²)'] +
    0.1 * synthetic_data['Operating Temperature (°C)'] -
    0.05 * synthetic_data['Hydrogen Flow Rate (mol/s)'] +
    0.03 * synthetic_data['Oxygen Flow Rate (mol/s)'] +
    0.01 * synthetic_data['Humidity of Feed Gases (%)'] -
    0.1 * synthetic_data['Membrane Thickness (μm)'] +
    0.05 * synthetic_data['Catalyst Loading (mg/cm²)'] -
    0.02 * synthetic_data['Membrane Swelling Ratio (%)'] +
    0.08 * synthetic_data['Anode Pt Loading (mg/cm²)'] +
    0.06 * synthetic_data['Cathode Pt Loading (mg/cm²)'] +
    0.04 * synthetic_data['Membrane Conductivity (S/cm)'] +
    0.03 * synthetic_data['Anode/Cathode Gas Diffusion Layer Thickness (μm)'] -
    0.02 * synthetic_data['Anode/Cathode Porosity (%)'] -
    0.03 * synthetic_data['Membrane Equivalent Weight (g/mol)'] +
    0.05 * synthetic_data['Anode/Cathode Loading Ratio'] +
    0.02 * synthetic_data['Cell Active Area (cm²)'] +
    0.03 * synthetic_data['Cell Pressure (atm)'] -
    0.01 * synthetic_data['Gas Humidification Temperature (°C)']
)

# Combine parameters and power output into a feature matrix
features = np.column_stack([synthetic_data[param] for param in param_ranges.keys()])
target = power_output


# # Data Splitting and Normalization
# 

# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalize input features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# #  Model Definition and Training
# 
# 

# In[3]:


import tensorflow as tf

# Build a deeper DNN model with batch normalization and dropout
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=150, batch_size=64)


# # Model Evaluation and Dataframe Creation

# In[4]:


from sklearn.metrics import mean_squared_error
import pandas as pd

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Create a DataFrame with the synthetic dataset
synthetic_data['Power Output (W/cm²)'] = power_output
df = pd.DataFrame(synthetic_data)

# Save the synthetic dataset to a CSV file
df.to_csv('synthetic_fuel_cell_dataset_large.csv', index=False)


# # Plotting Graphs

# In[6]:


import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot actual vs. predicted power output
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred)
plt.title('Actual vs. Predicted Power Output')
plt.xlabel('Actual Power Output')
plt.ylabel('Predicted Power Output')
plt.show()

# Plot a histogram of the residuals
residuals = y_test - y_pred.flatten()
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=50)
plt.title('Residuals Histogram')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Plot a scatter plot of the residuals
plt.figure(figsize=(8, 4))
plt.scatter(y_test, residuals)
plt.title('Actual Power Output vs. Residuals')
plt.xlabel('Actual Power Output')
plt.ylabel('Residuals')
plt.show()



# # conclusion 6

# In[8]:


print("Cell 6: Project Overview and Explanations\n")

# Project Overview
print("Project Overview:\n"
      "This project is dedicated to simulating and predicting fuel cell performance, with a particular focus on its potential significance for upcoming electric vehicle (EV) generation. By generating a synthetic dataset that encompasses various fuel cell performance parameters, we aim to optimize fuel cell design and contribute to the efficiency of electric vehicles. This endeavor aligns with the broader goal of advancing cleaner and more sustainable energy conversion technologies.\n")

# Explanation for Each Cell
print("Explanation for Each Cell:\n"
      "In Cell 1, we set the foundation by defining parameter ranges that impact fuel cell behavior. These parameters include cell voltage, current density, operating temperature, and many more. Through this data, we create a simulated environment, capturing the interactions between these parameters. The power output, a crucial performance metric, is calculated through a simplified equation.\n\n"

      "Cell 2 further prepares the data for analysis by splitting it into training and testing sets. This division ensures the model's ability to generalize beyond the data it was trained on. To guarantee accurate model training, we normalize the input features using MinMaxScaler, which scales them to a common range. This step is essential for a neural network to converge effectively.\n\n"

      "Cell 3 is dedicated to constructing the neural network model itself. We design a deep neural network architecture that encompasses various layers, including dense layers, batch normalization, and dropout layers. The architecture is chosen to account for the complexity of the problem and prevent overfitting. After compiling the model with an appropriate loss function, we train it using the training data to learn the relationships between the input parameters and power output.\n\n"

      "In Cell 4, we evaluate the trained model's performance. By predicting power output using the testing data and calculating the mean squared error, we quantify how well the model can generalize to new, unseen data. Additionally, we construct a DataFrame that combines the synthetic dataset with the simulated power output. This DataFrame is then saved as a CSV file for future reference or analysis.\n\n"

      "Cell 5 extends the analysis by visualizing the results. We generate graphs that showcase the model's learning progress by plotting the training and validation loss values. Moreover, we visualize the model's predictive abilities by plotting the actual power output against the predicted power output. We analyze the residuals—differences between actual and predicted values—through histograms and scatter plots to gain insights into the model's performance and the significance of different input parameters.\n\n"

      "In conclusion, this project offers a comprehensive exploration of fuel cell behavior, impacting the design of energy-efficient electric vehicles. The simulations, predictions, and analyses carried out in this project contribute to a cleaner energy future while advancing our understanding of fuel cell technologies.")


# In[ ]:




