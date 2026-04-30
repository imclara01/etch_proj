import pandas as pd
import numpy as np

def check_std():
    test_df = pd.read_csv('data/test_tstr.csv')
    test_df.columns = test_df.columns.str.strip()
    
    sensor = 'RF Btm Rfl Pwr'
    normal = test_df[test_df['Fault_Name'] == 'Normal']
    
    print(f"--- Sensor: {sensor} (Normal) ---")
    print(f"Mean: {normal[sensor].mean():.4f}")
    print(f"Std:  {normal[sensor].std():.4f}")
    print(f"Relative Std: {normal[sensor].std()/normal[sensor].mean()*100:.2f}%")

if __name__ == '__main__':
    check_std()
