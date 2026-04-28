import pandas as pd
import os

files = ['MACHINE_integrated.csv', 'RFM_integrated.csv']
data_dir = '/Users/crmoon/Desktop/proj/data'

for file in files:
    path = os.path.join(data_dir, file)
    print(f"\n--- Cycle Analysis for {file} ---")
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        
        # Identify columns
        time_step_col = 'Time_Step'
        time_col = 'Time' if 'Time' in df.columns else 'TIME'
        run_col = 'Run_Name'
        
        if time_step_col in df.columns and time_col in df.columns:
            # Create cycle_id: increments every time Time_Step is 1
            df['cycle_id'] = (df[time_step_col] == 1).cumsum()
            
            # Calculate metrics per cycle
            # We also include Run_Name to keep track of which run it belongs to
            cycle_stats = df.groupby(['cycle_id', run_col])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
            cycle_stats['duration'] = cycle_stats['max'] - cycle_stats['min']
            
            print(f"Total cycles identified: {len(cycle_stats)}")
            print("\nFirst 5 cycles average time (mean of Time column):")
            print(cycle_stats[['Run_Name', 'mean']].head())
            
            print("\nAverage of 'mean time' across all cycles:")
            print(cycle_stats['mean'].mean())
            
            print("\nAverage of 'cycle duration' across all cycles:")
            print(cycle_stats['duration'].mean())
            
        else:
            print(f"Required columns not found. Found columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
