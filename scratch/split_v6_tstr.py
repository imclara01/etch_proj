import pandas as pd
import numpy as np
import os
import re

def get_run_id(name):
    match = re.search(r'(\d+)', str(name))
    return match.group(1) if match else str(name)

def load_and_align_real():
    print("🚀 [Real Data] Loading and aligning MACHINE, OES, RFM...")
    oes = pd.read_csv('data/OES_integrated.csv')
    mach = pd.read_csv('data/MACHINE_integrated.csv')
    rfm = pd.read_csv('data/RFM_integrated.csv')
    
    oes.columns = oes.columns.str.strip()
    mach.columns = mach.columns.str.strip()
    rfm.columns = rfm.columns.str.strip()
    
    oes['run_id'] = oes['Run_Name'].apply(get_run_id)
    mach['run_id'] = mach['Run_Name'].apply(get_run_id)
    rfm['run_id'] = rfm['Run_Name'].apply(get_run_id)
    
    all_runs = sorted(oes['run_id'].unique())
    aligned_data = []
    
    o_feat = oes.columns.difference(['Data_Type', 'Run_Name', 'Time_Step', 'run_id'])
    m_feat = mach.columns.difference(['Data_Type', 'Run_Name', 'Time_Step', 'Time', 'Step Number', 'run_id'])
    r_feat = rfm.columns.difference(['Data_Type', 'Run_Name', 'Time_Step', 'TIME', 'run_id'])
    
    for rid in all_runs:
        o_sub = oes[oes['run_id'] == rid].sort_values('Time_Step')
        m_sub = mach[mach['run_id'] == rid].sort_values('Time_Step')
        r_sub = rfm[rfm['run_id'] == rid].sort_values('Time_Step')
        
        if len(m_sub) == 0 or len(r_sub) == 0: continue
            
        o_prog = np.linspace(0, 1, len(o_sub))
        m_prog = np.linspace(0, 1, len(m_sub))
        r_prog = np.linspace(0, 1, len(r_sub))
        
        m_interp = pd.DataFrame({col: np.interp(o_prog, m_prog, m_sub[col]) for col in m_feat})
        r_interp = pd.DataFrame({col: np.interp(o_prog, r_prog, r_sub[col]) for col in r_feat})
        
        combined = pd.concat([o_sub.reset_index(drop=True), m_interp.reset_index(drop=True), r_interp.reset_index(drop=True)], axis=1)
        aligned_data.append(combined)
        
    real_df = pd.concat(aligned_data, ignore_index=True)
    real_df['Is_Synthetic'] = 0
    return real_df

def apply_labels(df):
    print("🏷️  Applying labels...")
    ref = pd.read_csv('data/Augmented_Sensor_Data_v4.csv')
    ref_orig = ref[ref['Is_Synthetic'] == 0]
    label_map = {}
    for _, row in ref_orig[['Run_Name', 'Fault_Name']].drop_duplicates().iterrows():
        try:
            rid = str(int(float(row['Run_Name'])))
            if rid.startswith('3'):
                label_map[str(int(rid)-200)] = row['Fault_Name']
            label_map[rid] = row['Fault_Name']
        except: label_map[str(row['Run_Name'])] = row['Fault_Name']
    df['run_id'] = df['Run_Name'].apply(get_run_id)
    df['Fault_Name'] = df['run_id'].map(label_map).fillna('Normal')
    return df

def split_v6_tstr():
    print("🚀 Splitting v6 into Training set...")
    v6 = pd.read_csv('data/Augmented_Sensor_Data_v6.csv')
    v6.columns = v6.columns.str.strip()
    
    # Train: Synthetic from v6
    train_df = v6[v6['Is_Synthetic'] == 1].copy()
    
    # Add Synthetic Normal
    real_normal = v6[(v6['Is_Synthetic'] == 0) & (v6['Fault_Name'] == 'Normal')].copy()
    num_cols = real_normal.select_dtypes(include=['number']).columns.difference(['Is_Synthetic', 'Time_Step'])
    noise = np.random.normal(0, 0.01, (len(real_normal), len(num_cols)))
    synth_norm = real_normal.copy()
    synth_norm[num_cols] *= (1 + noise)
    synth_norm['Is_Synthetic'] = 1
    synth_norm['Run_Name'] = synth_norm['Run_Name'].astype(str) + "_SN"
    
    train_df = pd.concat([train_df, synth_norm], ignore_index=True)
    train_df.to_csv('data/train_tstr.csv', index=False)
    
    # Test: Real from Integrated
    test_df = load_and_align_real()
    test_df = apply_labels(test_df)
    test_df.to_csv('data/test_tstr.csv', index=False)
    
    print(f"✅ Created train_tstr.csv ({len(train_df)}) and test_tstr.csv ({len(test_df)})")

if __name__ == '__main__':
    split_v6_tstr()
