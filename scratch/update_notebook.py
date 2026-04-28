import json
import os
import textwrap

notebook_path = '3.AI agent_tool.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the new code blocks with explicit dedent to ensure clean indentation
partition_code = textwrap.dedent("""\
    # 데이터 분할 코드
    import pandas as pd
    import numpy as np
    import re
    import os

    def create_partitioned_simulation_data():
        mach_df = pd.read_csv('data/MACHINE_integrated.csv')
        oes_df = pd.read_csv('data/OES_integrated.csv')
        rfm_df = pd.read_csv('data/RFM_integrated.csv')

        for df in [mach_df, oes_df, rfm_df]:
            df.columns = df.columns.str.strip()

        def extract_id(run_name):
            match = re.search(r'(\\d+)', str(run_name))
            return match.group(1) if match else str(run_name)

        from collections import defaultdict
        mach_runs = defaultdict(list)
        for r in mach_df['Run_Name'].unique(): mach_runs[extract_id(r)].append(r)
        oes_runs = defaultdict(list)
        for r in oes_df['Run_Name'].unique(): oes_runs[extract_id(r)].append(r)
        rfm_runs = defaultdict(list)
        for r in rfm_df['Run_Name'].unique(): rfm_runs[extract_id(r)].append(r)

        common_ids = sorted(list(set(mach_runs.keys()) & set(oes_runs.keys()) & set(rfm_runs.keys())))
        print(f"✅ 공통 사이클 확인: {len(common_ids)}개")

        drop_cols = ['Fault_Name', 'Data_Type', 'Is_Synthetic', 'Synthesis_Method']
        def clean_df(df):
            to_drop = [c for c in drop_cols if c in df.columns]
            return df.drop(columns=to_drop)

        np.random.seed(42)
        np.random.shuffle(common_ids)
        
        id_groups = np.array_split(common_ids, 10)
        eq_ids = [f"EQ_{i:02d}" for i in range(1, 11)]
        final_frames = {'MACHINE': [], 'OES': [], 'RFM': []}

        for i, eq_id in enumerate(eq_ids):
            assigned_ids = id_groups[i]
            m_runs = [r for cid in assigned_ids for r in mach_runs[cid]]
            o_runs = [r for cid in assigned_ids for r in oes_runs[cid]]
            r_runs = [r for cid in assigned_ids for r in rfm_runs[cid]]

            m_part = clean_df(mach_df[mach_df['Run_Name'].isin(m_runs)].copy())
            o_part = clean_df(oes_df[oes_df['Run_Name'].isin(o_runs)].copy())
            r_part = clean_df(rfm_df[rfm_df['Run_Name'].isin(r_runs)].copy())

            for part, key in zip([m_part, o_part, r_part], ['MACHINE', 'OES', 'RFM']):
                part.insert(0, 'equipment_id', eq_id)
                final_frames[key].append(part)

        if not os.path.exists('data'): os.makedirs('data')
        for key in ['MACHINE', 'OES', 'RFM']:
            output_name = f"data/{key}_10EQ_Partitioned.csv"
            pd.concat(final_frames[key], ignore_index=True).to_csv(output_name, index=False)
            print(f"💾 생성 완료: {output_name}")

    if __name__ == '__main__':
        create_partitioned_simulation_data()
    """)

producer_code = textwrap.dedent("""\
    import pandas as pd
    import json
    import time
    from confluent_kafka import Producer

    producer = Producer({'bootstrap.servers': 'localhost:9092'})

    df_mach = pd.read_csv('data/MACHINE_10EQ_Partitioned.csv')
    df_rfm = pd.read_csv('data/RFM_10EQ_Partitioned.csv')
    df_oes = pd.read_csv('data/OES_10EQ_Partitioned.csv')

    def run_mixed_simulation():
        eq_ids = [f"EQ_{i:02d}" for i in range(1, 11)]
        machine_data_map = {eid: df_mach[df_mach['equipment_id'] == eid] for eid in eq_ids}
        rfm_data_map = {eid: df_rfm[df_rfm['equipment_id'] == eid] for eid in eq_ids}
        oes_data_map = {eid: df_oes[df_oes['equipment_id'] == eid] for eid in eq_ids}

        pointers = {eid: {'machine': 0, 'rfm': 0, 'oes': 0} for eid in eq_ids}
        step = 1
        print(f"🚀 10대 장비 동시 스트리밍 시작...")

        while True:
            any_data_sent = False
            for eid in eq_ids:
                for s_type, data_map in [('machine', machine_data_map), ('rfm', rfm_data_map), ('oes', oes_data_map)]:
                    curr_idx = pointers[eid][s_type]
                    if curr_idx < len(data_map[eid]):
                        row = data_map[eid].iloc[curr_idx]
                        payload = {
                            "equipment_id": eid,
                            "type": s_type,
                            "metrics": row.drop(['equipment_id', 'Run_Name']).to_dict(),
                            "timestamp": time.time(),
                            "run_name": row['Run_Name']
                        }
                        producer.produce('etching-data', key=eid, value=json.dumps(payload))
                        pointers[eid][s_type] += 1
                        any_data_sent = True
            
            if not any_data_sent: break
            producer.flush()
            if step % 100 == 0 or step == 1:
                print(f"[{time.strftime('%H:%M:%S')}] Step {step} - 스트리밍 진행 중...")
            step += 1
            time.sleep(0.01)

        print("🏁 스트리밍 완료")

    if __name__ == '__main__':
        run_mixed_simulation()
    """)

consumer_code = textwrap.dedent("""\
    import json
    import numpy as np
    import time
    from confluent_kafka import Consumer
    from collections import deque

    ALL_FEATURES = ['250.0', '261.8', '266.6', '272.2', '278.3', '284.6', '288.25', '308.25', '309.25', '324.8', '327.5', '336.98', '364.33', '388.0', '394.4', '395.8', '415.0', '532.6', '544.2', '556.7', '580.0', '611.5', '613.9', '616.3', '618.5', '639.7', '643.2', '644.9', '652.8', '660.0', '669.5', '670.6', '674.0', '725.0', '740.8', '748.5', '753.7', '770.6', '773.2', '781.0', '783.5', '787.5', '791.5', '250.0.1', '261.8.1', '266.6.1', '272.2.1', '278.3.1', '284.6.1', '288.25.1', '308.25.1', '309.25.1', '324.8.1', '327.5.1', '336.98.1', '364.33.1', '388.0.1', '394.4.1', '395.8.1', '415.0.1', '532.6.1', '544.2.1', '556.7.1', '580.0.1', '611.5.1', '613.9.1', '616.3.1', '618.5.1', '639.7.1', '643.2.1', '644.9.1', '652.8.1', '660.0.1', '669.5.1', '670.6.1', '674.0.1', '725.0.1', '740.8.1', '748.5.1', '753.7.1', '770.6.1', '773.2.1', '781.0.1', '783.5.1', '787.5.1', '791.5.1', '250.0.2', '261.8.2', '266.6.2', '272.2.2', '278.3.2', '284.6.2', '288.25.2', '308.25.2', '309.25.2', '324.8.2', '327.5.2', '336.98.2', '364.33.2', '388.0.2', '394.4.2', '395.8.2', '415.0.2', '532.6.2', '544.2.2', '556.7.2', '580.0.2', '611.5.2', '613.9.2', '616.3.2', '618.5.2', '639.7.2', '643.2.2', '644.9.2', '652.8.2', '660.0.2', '669.5.2', '670.6.2', '674.0.2', '725.0.2', '740.8.2', '748.5.2', '753.7.2', '770.6.2', '773.2.2', '781.0.2', '783.5.2', '787.5.2', '791.5.2', 'TIME', 'S1V1', 'S1V2', 'S1V3', 'S1V4', 'S1V5', 'S1I1', 'S1I2', 'S1I3', 'S1I4', 'S1I5', 'S1P1', 'S1P2', 'S1P3', 'S1P4', 'S1P5', 'S2V1', 'S2V2', 'S2V3', 'S2V4', 'S2V5', 'S2I1', 'S2I2', 'S2I3', 'S2I4', 'S2I5', 'S2P1', 'S2P2', 'S2P3', 'S2P4', 'S2P5', 'S3V1', 'S3V2', 'S3V3', 'S3V4', 'S3V5', 'S4V1', 'S4V2', 'S4V3', 'S4V4', 'S4V5', 'S34PV1', 'S34PV2', 'S34PV3', 'S34PV4', 'S34PV5', 'S3I1', 'S3I2', 'S3I3', 'S3I4', 'S3I5', 'S4I1', 'S4I2', 'S4I3', 'S4I4', 'S4I5', 'S34PI1', 'S34PI2', 'S34PI3', 'S34PI4', 'S34PI5', 'S34V1', 'S34V2', 'S34V3', 'S34V4', 'S34V5', 'S34I1', 'S34I2', 'S34I3', 'S34I4', 'S34I5', 'Time.1', 'Step Number', 'BCl3 Flow', 'Cl2 Flow', 'RF Btm Pwr', 'RF Btm Rfl Pwr', 'Endpt A', 'He Press', 'Pressure', 'RF Tuner', 'RF Load', 'RF Phase Err', 'RF Pwr', 'RF Impedance', 'TCP Tuner', 'TCP Phase Err', 'TCP Impedance', 'TCP Top Pwr', 'TCP Rfl Pwr', 'TCP Load', 'Vat Valve']

    class LSTMAgent:
        def __init__(self, eq_id, window_size=30):
            self.eq_id = eq_id
            self.window_size = window_size
            self.latest_cache = {feat: 0.0 for feat in ALL_FEATURES}
            self.buffer = deque(maxlen=window_size)
            self.current_run = None

        def update(self, sensor_type, metrics, run_name):
            if run_name != self.current_run:
                print(f"\\n[{self.eq_id}] New Run: {run_name}. Resetting.")
                self.current_run = run_name
                self.buffer.clear()
                self.latest_cache = {feat: 0.0 for feat in ALL_FEATURES}

            for k, v in metrics.items():
                if k in self.latest_cache: self.latest_cache[k] = v
            
            if sensor_type == 'oes':
                print('*', end='', flush=True)
                row = [self.latest_cache[f] for f in ALL_FEATURES]
                self.buffer.append(row)
                if len(self.buffer) == self.window_size:
                    return np.array(self.buffer).reshape(1, self.window_size, -1)
            else:
                print('.', end='', flush=True)
            return None

    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'etching_analysis_group_v3',
        'auto.offset.reset': 'latest'
    }
    consumer = Consumer(conf)
    consumer.subscribe(['etching-data'])

    agents = {f"EQ_{i:02d}": LSTMAgent(f"EQ_{i:02d}") for i in range(1, 11)}

    print("🧠 LSTM 실시간 이상 탐지 에이전트 가동 중...")

    try:
        while True:
            msg = consumer.poll(0.1)
            if msg is None: continue
            if msg.error(): continue

            try:
                val = msg.value().decode('utf-8')
                if not val: continue
                data = json.loads(val)
                
                eq_id = data['equipment_id']
                agent = agents[eq_id]
                input_tensor = agent.update(data['type'], data['metrics'], data['run_name'])
                
                if input_tensor is not None:
                    print(f"\\n[{eq_id}] 윈도우 완성! (Shape: {input_tensor.shape})")
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    except KeyboardInterrupt:
        print("중단됨")
    """)

# Update cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'create_partitioned_simulation_data' in source:
            cell['source'] = [line + '\n' for line in partition_code.split('\n')]
            cell['source'][-1] = cell['source'][-1].strip()
        elif 'run_mixed_simulation' in source:
            cell['source'] = [line + '\n' for line in producer_code.split('\n')]
            cell['source'][-1] = cell['source'][-1].strip()
        elif 'class LSTMAgent' in source:
            cell['source'] = [line + '\n' for line in consumer_code.split('\n')]
            cell['source'][-1] = cell['source'][-1].strip()

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated with fixed indentation.")
