import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

# Define the columns for the DataFrame
columns = ["date", "joint_angle", "joint_velocity", "joint_torque", "intervention", "costs", "technicians"]

# Generate synthetic data
data = []
for _ in range(300):
    date = fake.date_between(start_date='-2y', end_date='today')
    joint_angle = np.round(np.random.uniform(-180, 180), 2)
    joint_velocity = np.round(np.random.uniform(-100, 100), 2)
    joint_torque = np.round(np.random.uniform(-50, 50), 2)
    
    # Generate detailed intervention notes
    intervention = fake.paragraph(nb_sentences=3, variable_nb_sentences=True)
    
    # Generate costs
    costs = np.round(np.random.uniform(100, 5000), 2)
    
    # Generate technician names
    technicians = ", ".join([fake.name() for _ in range(np.random.randint(1, 3))])
    
    data.append([date, joint_angle, joint_velocity, joint_torque, intervention, costs, technicians])

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to CSV
file_path = 'robotic_arm_maintenance_records_detailed.csv'
df.to_csv(file_path, index=False)
