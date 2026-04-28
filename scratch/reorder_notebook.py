import json

file_path = '/Users/crmoon/Desktop/proj/3.AI agent_tool.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Define markers for each unique block
markers = {
    'intro': '# 3. AI 에이전트의 도구 생성',
    'arch': '## 프로젝트 아키텍처 구현',
    'kafka_setup': '### 1. Kafka Consumer',
    'data_prep_md': '### 2. 카프카 쏴줄 데이터 준비',
    'data_prep_code': 'create_partitioned_simulation_data()',
    'consumer_md': '### 3. 실시간 데이터 정렬',
    'consumer_code': 'LSTMAgent',
    'producer_md': '[Producer] 10대 장비 실시간 스트리밍 코드',
    'producer_code': 'run_mixed_simulation()',
    'modeling_md': '### 4. Keras(텐서플로우) LSTM 모델링'
}

unique_cells = []
seen_content = set()

for cell in cells:
    content = "".join(cell.get('source', []))
    if content not in seen_content:
        unique_cells.append(cell)
        seen_content.add(content)

# Group unique cells into logical blocks
intro_block = []
data_prep_block = []
consumer_block = []
producer_block = []
modeling_block = []

current_block = intro_block

for cell in unique_cells:
    source = "".join(cell.get('source', []))
    
    if markers['data_prep_md'] in source:
        current_block = data_prep_block
    elif markers['consumer_md'] in source:
        current_block = consumer_block
    elif markers['producer_md'] in source:
        current_block = producer_block
    elif markers['modeling_md'] in source:
        current_block = modeling_block
    
    current_block.append(cell)

# Final desired order: Intro -> Data Prep -> Consumer -> Producer -> Modeling
final_cells = intro_block + data_prep_block + consumer_block + producer_block + modeling_block

nb['cells'] = final_cells

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook cleanup and reordering complete.")
print(f"Total cells: {len(final_cells)}")
