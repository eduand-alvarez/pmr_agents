import pandas as pd
from xgboost import XGBClassifier

# Load the trained model
model = XGBClassifier()
model.load_model('xgboost_model.json')

# Function to predict maintenance requirement
def predict_maintenance(sensor_data):
    """
    Predict whether maintenance is required based on sensor data.

    Parameters:
    sensor_data (list or np.array): Sensor readings (e.g., [joint_angle, joint_velocity, joint_torque])

    Returns:
    int: 1 if maintenance is required, 0 otherwise
    """
    sensor_data = pd.DataFrame([sensor_data], columns=['joint_angle', 'joint_velocity', 'joint_torque'])
    prediction = model.predict(sensor_data)
    return int(prediction[0])

# Example usage
sensor_readings = [0.5, 0.1, -0.3]  # Replace with actual sensor readings
prediction = predict_maintenance(sensor_readings)
print(f'Maintenance required: {prediction}')
