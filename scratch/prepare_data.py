import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
FDATA_DIR = 'data'
OES_FILE = os.path.join(FDATA_DIR, 'OES_integrated.csv')
MACH_FILE = os.path.join(FDATA_DIR, 'MACHINE_integrated.csv')
RFM_FILE = os.path.join(FDATA_DIR, 'RFM_integrated.csv')
REF_FILE = 'data/Augmented_Sensor_Data_v4.csv'

# Split Params
TRAIN_NORMAL = 70
VAL_NORMAL = 10
TEST_NORMAL = 23 # Total normal is around 104

TRAIN_FAULT = 10
VAL_FAULT = 5
TEST_FAULT = 5 # Total fault is 20 in v4. I'll use 5 for Test to match 20 total.

# Augmentation Params
AUG_MULTIPLIER = 7 # Each method produces 7 runs. Total 7+7+7 = 21x.

def get_run_id(name):
    import re
    match = re.search(r'(\d+)', str(name))
    return match.group(1) if match else str(name)

def load_and_align():
    print("Loading data...")
    oes = pd.read_csv(OES_FILE)
    mach = pd.read_csv(MACH_FILE)
    rfm = pd.read_csv(RFM_FILE)
    
    # Standardize Run Names
    oes['run_id'] = oes['Run_Name'].apply(get_run_id)
    mach['run_id'] = mach['Run_Name'].apply(get_run_id)
    rfm['run_id'] = rfm['Run_Name'].apply(get_run_id)
    
    print("Aligning runs...")
    all_runs = sorted(oes['run_id'].unique())
    aligned_data = []
    
    for rid in all_runs:
        o_sub = oes[oes['run_id'] == rid].sort_values('Time_Step')
        m_sub = mach[mach['run_id'] == rid].sort_values('Time_Step')
        r_sub = rfm[rfm['run_id'] == rid].sort_values('Time_Step')
        
        if len(m_sub) == 0 or len(r_sub) == 0:
            continue
            
        o_prog = np.linspace(0, 1, len(o_sub))
        m_prog = np.linspace(0, 1, len(m_sub))
        r_prog = np.linspace(0, 1, len(r_sub))
        
        m_feat = m_sub.select_dtypes(include=[np.number]).columns.difference(['Time_Step', 'Time', 'Step Number'])
        r_feat = r_sub.select_dtypes(include=[np.number]).columns.difference(['Time_Step'])
        
        m_interp = pd.DataFrame({col: np.interp(o_prog, m_prog, m_sub[col]) for col in m_feat})
        r_interp = pd.DataFrame({col: np.interp(o_prog, r_prog, r_sub[col]) for col in r_feat})
        
        combined = pd.concat([
            o_sub.reset_index(drop=True),
            m_interp.reset_index(drop=True),
            r_interp.reset_index(drop=True)
        ], axis=1)
        
        aligned_data.append(combined)
        
    return pd.concat(aligned_data, ignore_index=True)

def apply_labels(df):
    print("Applying labels from reference...")
    ref = pd.read_csv(REF_FILE)
    ref_orig = ref[ref['Is_Synthetic'] == 0]
    
    label_map = {}
    ref_runs = ref_orig[['Run_Name', 'Fault_Name']].drop_duplicates()
    
    for _, row in ref_runs.iterrows():
        rid = str(int(row['Run_Name']))
        if rid.startswith('3'):
            alt_rid = str(int(rid) - 200)
            label_map[alt_rid] = row['Fault_Name']
        label_map[rid] = row['Fault_Name']
        
    df['Fault_Name'] = df['run_id'].map(label_map).fillna('Unknown')
    return df

def split_data(df):
    print("Splitting data...")
    runs = df[['run_id', 'Fault_Name']].drop_duplicates()
    
    normal_runs = runs[runs['Fault_Name'] == 'Normal']['run_id'].tolist()
    fault_runs = runs[(runs['Fault_Name'] != 'Normal') & (runs['Fault_Name'] != 'Unknown')]['run_id'].tolist()
    
    np.random.seed(42)
    np.random.shuffle(normal_runs)
    np.random.shuffle(fault_runs)
    
    train_n = normal_runs[:TRAIN_NORMAL]
    val_n = normal_runs[TRAIN_NORMAL:TRAIN_NORMAL+VAL_NORMAL]
    test_n = normal_runs[TRAIN_NORMAL+VAL_NORMAL:TRAIN_NORMAL+VAL_NORMAL+TEST_NORMAL]
    
    train_f = fault_runs[:TRAIN_FAULT]
    val_f = fault_runs[TRAIN_FAULT:TRAIN_FAULT+VAL_FAULT]
    test_f = fault_runs[TRAIN_FAULT+VAL_FAULT:]
    
    return train_n, train_f, val_n, val_f, test_n, test_f

def augment_pg_ha(run_data, fault_name):
    main_sensors = {
        'Pr': ['Pressure', 'Vat Valve'],
        'Cl2': ['Cl2 Flow'],
        'BCl3': ['BCl3 Flow'],
        'TCP': ['TCP Top Pwr', 'TCP Load'],
        'RF': ['RF Btm Pwr', 'RF Impedance'],
        'He Chuck': ['He Press']
    }
    target_group = None
    for group in main_sensors:
        if group in fault_name:
            target_group = group
            break
            
    augmented_runs = []
    sensor_cols = [c for c in run_data.columns if any(s in str(c) for s in ['Flow', 'Pwr', 'Press', 'Pressure', 'Valve', 'Load', 'Impedance'])]
    
    for i in range(AUG_MULTIPLIER):
        new_run = run_data.copy()
        scale_main = 1.0 + (np.random.rand() - 0.5) * 0.1
        scale_others = 1.0 + (np.random.rand() - 0.5) * 0.02
        
        for col in sensor_cols:
            is_main = False
            if target_group:
                if any(s in col for s in main_sensors[target_group]):
                    is_main = True
            if is_main:
                new_run[col] = new_run[col] * scale_main
            else:
                new_run[col] = new_run[col] * scale_others
        
        new_run['Is_Synthetic'] = 1
        new_run['Synthesis_Method'] = 'PGHA'
        new_run['Run_Name'] = f"{run_data['Run_Name'].iloc[0]}_PGHA_{i}"
        augmented_runs.append(new_run)
    return augmented_runs

def augment_jittering(run_data):
    augmented_runs = []
    num_cols = run_data.select_dtypes(include=[np.number]).columns.difference(['Time_Step', 'Time', 'Step Number', 'Is_Synthetic'])
    for i in range(AUG_MULTIPLIER):
        new_run = run_data.copy()
        std = 0.01 + np.random.rand() * 0.04
        noise = np.random.normal(0, std, (len(new_run), len(num_cols)))
        new_run[num_cols] = new_run[num_cols] + noise
        new_run['Is_Synthetic'] = 1
        new_run['Synthesis_Method'] = 'Jittering'
        new_run['Run_Name'] = f"{run_data['Run_Name'].iloc[0]}_Jitter_{i}"
        augmented_runs.append(new_run)
    return augmented_runs

def augment_mag_warping(run_data):
    augmented_runs = []
    num_cols = run_data.select_dtypes(include=[np.number]).columns.difference(['Time_Step', 'Time', 'Step Number', 'Is_Synthetic'])
    for i in range(AUG_MULTIPLIER):
        new_run = run_data.copy()
        scale = 0.95 + np.random.rand() * 0.1
        new_run[num_cols] = new_run[num_cols] * scale
        new_run['Is_Synthetic'] = 1
        new_run['Synthesis_Method'] = 'MagWarping'
        new_run['Run_Name'] = f"{run_data['Run_Name'].iloc[0]}_MagWarp_{i}"
        augmented_runs.append(new_run)
    return augmented_runs

def run_augmentation(df, train_f):
    print("Performing augmentation...")
    train_fault_df = df[df['run_id'].isin(train_f)].copy()
    train_fault_df['Is_Synthetic'] = 0
    train_fault_df['Synthesis_Method'] = 'Original'
    augmented_list = [train_fault_df]
    for rid in train_f:
        run_data = train_fault_df[train_fault_df['run_id'] == rid]
        fname = run_data['Fault_Name'].iloc[0]
        augmented_list.extend(augment_pg_ha(run_data, fname))
        augmented_list.extend(augment_jittering(run_data))
        augmented_list.extend(augment_mag_warping(run_data))
    return pd.concat(augmented_list, ignore_index=True)

def generate_report(orig_df, aug_df):
    print("Generating report...")
    features = [c for c in orig_df.columns if isinstance(c, str) and ('.' in c or any(s in c for s in ['Flow', 'Pwr', 'Press', 'Pressure', 'Valve', 'Load', 'Impedance']))]
    plot_data = pd.concat([orig_df.iloc[::10], aug_df.iloc[::10]])
    X = plot_data[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 7))
    plt.scatter(components[len(orig_df)//10:, 0], components[len(orig_df)//10:, 1], alpha=0.2, label='Augmented', c='orange', zorder=1)
    plt.scatter(components[:len(orig_df)//10, 0], components[:len(orig_df)//10, 1], alpha=0.8, label='Original', c='blue', zorder=2)
    plt.title('PCA Distribution: Original vs Augmented')
    plt.legend()
    plt.savefig('augmentation_pca_report.png')

def main():
    df = load_and_align()
    df = apply_labels(df)
    train_n, train_f, val_n, val_f, test_n, test_f = split_data(df)
    
    train_normal_df = df[df['run_id'].isin(train_n)].copy()
    train_normal_df['Is_Synthetic'] = 0
    train_normal_df['Synthesis_Method'] = 'Original'
    
    train_fault_aug_df = run_augmentation(df, train_f)
    train_final = pd.concat([train_normal_df, train_fault_aug_df], ignore_index=True)
    
    val_final = df[df['run_id'].isin(val_n + val_f)].copy()
    test_final = df[df['run_id'].isin(test_n + test_f)].copy()
    
    train_final.to_csv('data/train_augmented.csv', index=False)
    val_final.to_csv('data/val_split.csv', index=False)
    test_final.to_csv('data/test_split.csv', index=False)
    
    generate_report(df[df['run_id'].isin(train_f)], train_fault_aug_df)
    print("Success.")

if __name__ == "__main__":
    main()
