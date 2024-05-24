import numpy as np
import pandas as pd
import time

def generate_synthetic_data(num_samples=10000):
    np.random.seed(42)
    data = []
    
    for _ in range(num_samples):
        timestamp = time.time()
        joint_angle = np.random.normal(loc=0.0, scale=1.0)
        joint_velocity = np.random.normal(loc=0.0, scale=1.0)
        joint_torque = np.random.normal(loc=0.0, scale=1.0)
        
        # Simulate an anomaly by introducing higher values for failure cases
        if np.random.rand() > 0.95:  # 5% chance of failure
            joint_angle += np.random.normal(loc=5.0, scale=1.0)
            joint_velocity += np.random.normal(loc=5.0, scale=1.0)
            joint_torque += np.random.normal(loc=5.0, scale=1.0)
            label = 1
        else:
            label = 0
        
        data.append([timestamp, joint_angle, joint_velocity, joint_torque, label])
    
    columns = ['timestamp', 'joint_angle', 'joint_velocity', 'joint_torque', 'label']
    df = pd.DataFrame(data, columns=columns)
    
    return df

# Generate the synthetic data
synthetic_data = generate_synthetic_data(num_samples=10000)
synthetic_data.to_csv('synthetic_robot_data.csv', index=False)
print(synthetic_data.head())
