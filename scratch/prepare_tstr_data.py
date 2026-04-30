import pandas as pd
import numpy as np
import os
import re

# --- Configuration ---
DATA_DIR = 'data'
OES_FILE = os.path.join(DATA_DIR, 'OES_integrated.csv')
MACH_FILE = os.path.join(DATA_DIR, 'MACHINE_integrated.csv')
RFM_FILE = os.path.join(DATA_DIR, 'RFM_integrated.csv')
V6_FILE = os.path.join(DATA_DIR, 'Augmented_Sensor_Data_v6.csv')
REF_FILE = os.path.join(DATA_DIR, 'Augmented_Sensor_Data_v4.csv')

def get_run_id(name):
    match = re.search(r'(\d+)', str(name))
    return match.group(1) if match else str(name)

def load_and_align_real():
    print("🚀 [Real Data] Loading and aligning MACHINE, OES, RFM...")
    oes = pd.read_csv(OES_FILE)
    mach = pd.read_csv(MACH_FILE)
    rfm = pd.read_csv(RFM_FILE)
    
    # [Fix] Strip whitespace from columns immediately
    oes.columns = oes.columns.str.strip()
    mach.columns = mach.columns.str.strip()
    rfm.columns = rfm.columns.str.strip()
    
    # Standardize Run IDs
    oes['run_id'] = oes['Run_Name'].apply(get_run_id)
    mach['run_id'] = mach['Run_Name'].apply(get_run_id)
    rfm['run_id'] = rfm['Run_Name'].apply(get_run_id)
    
    all_runs = sorted(oes['run_id'].unique())
    aligned_data = []
    
    # Identify feature columns
    o_feat = oes.columns.difference(['Data_Type', 'Run_Name', 'Time_Step', 'run_id'])
    m_feat = mach.columns.difference(['Data_Type', 'Run_Name', 'Time_Step', 'Time', 'Step Number', 'run_id'])
    r_feat = rfm.columns.difference(['Data_Type', 'Run_Name', 'Time_Step', 'TIME', 'run_id'])
    
    for rid in all_runs:
        o_sub = oes[oes['run_id'] == rid].sort_values('Time_Step')
        m_sub = mach[mach['run_id'] == rid].sort_values('Time_Step')
        r_sub = rfm[rfm['run_id'] == rid].sort_values('Time_Step')
        
        if len(m_sub) == 0 or len(r_sub) == 0:
            continue
            
        o_prog = np.linspace(0, 1, len(o_sub))
        m_prog = np.linspace(0, 1, len(m_sub))
        r_prog = np.linspace(0, 1, len(r_sub))
        
        # Interpolate MACHINE and RFM to OES time steps
        m_interp = pd.DataFrame({col: np.interp(o_prog, m_prog, m_sub[col]) for col in m_feat})
        r_interp = pd.DataFrame({col: np.interp(o_prog, r_prog, r_sub[col]) for col in r_feat})
        
        # Metadata from MACHINE
        m_meta = pd.DataFrame({
            'Time': np.interp(o_prog, m_prog, m_sub['Time']),
            'Step Number': np.interp(o_prog, m_prog, m_sub['Step Number'])
        })
        
        # Metadata from RFM
        r_meta = pd.DataFrame({
            'TIME': np.interp(o_prog, r_prog, r_sub['TIME'])
        })
        
        combined = pd.concat([
            o_sub.reset_index(drop=True),
            m_interp.reset_index(drop=True),
            r_interp.reset_index(drop=True),
            m_meta.reset_index(drop=True),
            r_meta.reset_index(drop=True)
        ], axis=1)
        
        aligned_data.append(combined)
        
    real_df = pd.concat(aligned_data, ignore_index=True)
    real_df['Is_Synthetic'] = 0
    real_df['Synthesis_Method'] = 'Real'
    return real_df

def apply_labels(df):
    print("🏷️  Applying labels from reference...")
    ref = pd.read_csv(REF_FILE)
    ref_orig = ref[ref['Is_Synthetic'] == 0]
    
    label_map = {}
    ref_runs = ref_orig[['Run_Name', 'Fault_Name']].drop_duplicates()
    
    for _, row in ref_runs.iterrows():
        try:
            rid = str(int(float(row['Run_Name'])))
            if rid.startswith('3'):
                alt_rid = str(int(rid) - 200)
                label_map[alt_rid] = row['Fault_Name']
            label_map[rid] = row['Fault_Name']
        except:
            label_map[str(row['Run_Name'])] = row['Fault_Name']
            
    df['Fault_Name'] = df['run_id'].map(label_map).fillna('Unknown')
    return df

def prepare_train_set():
    print("🧪 [Train Data] Loading and processing v6 for synthetic-only training...")
    v6 = pd.read_csv(V6_FILE)
    v6.columns = v6.columns.str.strip()
    
    # 1. Keep synthetic faults
    synth_faults = v6[(v6['Is_Synthetic'] == 1) & (v6['Fault_Name'] != 'Normal')].copy()
    
    # 2. Extract real normal and augment it to create "Synthetic Normal"
    real_normal = v6[v6['Fault_Name'] == 'Normal'].copy()
    
    print(f"   - Generating synthetic normal from {len(real_normal)} real rows...")
    # Simple jittering for synthetic normal
    num_cols = real_normal.select_dtypes(include=[np.number]).columns.difference(['Is_Synthetic', 'Time_Step', 'Time', 'Step Number'])
    
    synth_normal_list = []
    for i in range(2): # Create 2x synthetic normal
        noise_batch = real_normal.copy()
        noise = np.random.normal(0, 0.02, (len(noise_batch), len(num_cols)))
        noise_batch[num_cols] = noise_batch[num_cols] * (1 + noise)
        noise_batch['Is_Synthetic'] = 1
        noise_batch['Synthesis_Method'] = 'NormalJitter'
        noise_batch['Run_Name'] = noise_batch['Run_Name'].astype(str) + f"_SynthNorm_{i}"
        synth_normal_list.append(noise_batch)
            
    combined_synth_normal = pd.concat(synth_normal_list, ignore_index=True)
    train_df = pd.concat([synth_faults, combined_synth_normal], ignore_index=True)
    return train_df

def main():
    # 1. Prepare Training Data
    train_df = prepare_train_set()
    train_df.to_csv('data/train_tstr.csv', index=False)
    print(f"✅ Training set saved to data/train_tstr.csv ({len(train_df)} rows, 100% Synthetic)")
    
    # 2. Prepare Testing Data
    real_df = load_and_align_real()
    real_df = apply_labels(real_df)
    real_df.to_csv('data/test_tstr.csv', index=False)
    print(f"✅ Testing set saved to data/test_tstr.csv ({len(real_df)} rows, 100% Real)")

    # 3. Quick check for overlap
    train_runs = set(train_df['Run_Name'].unique())
    test_runs = set(real_df['Run_Name'].unique())
    overlap = train_runs.intersection(test_runs)
    print(f"🔍 Overlap check: {len(overlap)} runs overlap.")
    if len(overlap) > 0:
        print(f"⚠️  Warning: Overlapping runs found: {overlap}")

if __name__ == "__main__":
    main()
