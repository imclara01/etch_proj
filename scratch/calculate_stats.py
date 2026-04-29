import pandas as pd
import json
import os

def calculate_sensor_stats(data_path='data/train_augmented.csv', output_path='models/sensor_stats.json'):
    print(f"Reading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Filter for Normal data
    normal_df = df[df['Fault_Name'] == 'Normal']
    
    # Identify sensor features
    # Excluding non-sensor columns
    exclude = ['Time_Step', 'Time', 'Step Number', 'Run_Name', 'run_id', 'Fault_Name', 'Is_Synthetic', 'Synthesis_Method']
    features = [c for c in normal_df.columns if c not in exclude and normal_df[c].dtype in ['float64', 'int64']]
    
    stats = {}
    for feat in features:
        mean = normal_df[feat].mean()
        std = normal_df[feat].std()
        stats[feat] = {
            'mean': float(mean),
            'std': float(std),
            'min': float(normal_df[feat].min()),
            'max': float(normal_df[feat].max()),
            'lower_bound': float(mean - 3 * std),
            'upper_bound': float(mean + 3 * std)
        }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Sensor stats saved to {output_path}")

if __name__ == "__main__":
    calculate_sensor_stats()
