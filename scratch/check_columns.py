import pandas as pd

def check():
    oes = pd.read_csv('data/OES_integrated.csv')
    mach = pd.read_csv('data/MACHINE_integrated.csv')
    rfm = pd.read_csv('data/RFM_integrated.csv')
    synth = pd.read_csv('data/Augmented_Sensor_Data_v6.csv')
    
    print(f"OES columns: {len(oes.columns)}")
    print(f"MACHINE columns: {len(mach.columns)}")
    print(f"RFM columns: {len(rfm.columns)}")
    print(f"Synthetic columns: {len(synth.columns)}")
    
    # Check if synthetic columns contain all features
    exclude = ['Time_Step', 'Time', 'Step Number', 'Run_Name', 'run_id', 'Fault_Name', 'Is_Synthetic', 'Synthesis_Method', 'Data_Type', 'TIME', 'Time.1', 'TIME.1']
    synth_features = [c for c in synth.columns if c.strip() not in exclude]
    print(f"Synthetic feature count: {len(synth_features)}")
    
    # Check for duplicates in OES
    print(f"OES unique columns: {len(set(oes.columns))}")
    print(f"OES duplicate check: {oes.columns.value_counts().head(5)}")

if __name__ == '__main__':
    check()
