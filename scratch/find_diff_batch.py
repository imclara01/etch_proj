import pandas as pd
import numpy as np

def find_diff(fault_name):
    test_df = pd.read_csv('data/test_tstr.csv')
    test_df.columns = test_df.columns.str.strip()
    
    exclude = ['Time_Step', 'Time', 'Step Number', 'Run_Name', 'run_id', 'Fault_Name', 'Is_Synthetic', 'Synthesis_Method', 'Data_Type', 'TIME', 'Time.1', 'TIME.1']
    sensors = [c for c in test_df.columns if c not in exclude and test_df[c].dtype in ['float64', 'int64']]
    
    normal = test_df[test_df['Fault_Name'] == 'Normal']
    fault = test_df[test_df['Fault_Name'] == fault_name]
    
    if fault.empty:
        print(f"No samples for {fault_name}")
        return

    diffs = []
    for s in sensors:
        mn = normal[s].mean()
        mf = fault[s].mean()
        diff = abs(mf - mn) / (abs(mn) + 1e-9) * 100
        diffs.append({'Sensor': s, 'Diff%': diff})
        
    df_diff = pd.DataFrame(diffs).sort_values('Diff%', ascending=False)
    print(f"\nTop Diffs for {fault_name}:")
    print(df_diff.head(10))

if __name__ == '__main__':
    for f in ['BCl3 +5', 'He Chuck', 'Pr +3', 'TCP +50']:
        find_diff(f)
