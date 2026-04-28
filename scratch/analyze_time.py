import pandas as pd
import os

files = ['MACHINE_integrated.csv', 'RFM_integrated.csv']
data_dir = '/Users/crmoon/Desktop/proj/data'

for file in files:
    path = os.path.join(data_dir, file)
    print(f"\n--- Analysis for {file} ---")
    try:
        df = pd.read_csv(path)
        # Clean column names (strip spaces and handle potential encoding issues)
        df.columns = [c.strip() for c in df.columns]
        
        # Identify Run_Name and Time columns
        run_col = 'Run_Name'
        time_col = 'Time' if 'Time' in df.columns else 'TIME'
        
        if run_col in df.columns and time_col in df.columns:
            avg_time = df.groupby(run_col)[time_col].mean()
            print(avg_time)
        else:
            print(f"Required columns not found. Found columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
