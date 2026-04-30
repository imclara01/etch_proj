import pandas as pd
import numpy as np

def compare_values():
    train_df = pd.read_csv('data/train_tstr.csv')
    test_df = pd.read_csv('data/test_tstr.csv')
    
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    
    sensor = 'RF Btm Pwr'
    fault = 'RF +10'
    
    print(f"--- Sensor: {sensor} ---")
    
    tn = train_df[train_df['Fault_Name'] == 'Normal'][sensor].mean()
    tf = train_df[train_df['Fault_Name'] == fault][sensor].mean()
    print(f"Synthetic: Normal Mean={tn:.4f}, {fault} Mean={tf:.4f}, Diff={(tf-tn)/tn*100:.2f}%")
    
    rn = test_df[test_df['Fault_Name'] == 'Normal'][sensor].mean()
    rf = test_df[test_df['Fault_Name'] == fault][sensor].mean()
    print(f"Real:      Normal Mean={rn:.4f}, {fault} Mean={rf:.4f}, Diff={(rf-rn)/rn*100:.2f}%")

if __name__ == '__main__':
    compare_values()
