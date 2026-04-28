import pandas as pd
import json
import time
import os
from confluent_kafka import Producer

# --- Configuration ---
TOPIC = 'sensor-data-stream'
BOOTSTRAP_SERVERS = 'localhost:9092'
DATA_FILE = 'data/test_split.csv'

def delivery_report(err, msg):
    if err is not None:
        print(f"❌ Message delivery failed: {err}")
    else:
        # print(f"✅ Message delivered to {msg.topic()} [{msg.partition()}]")
        pass

def run_producer():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found: {DATA_FILE}")
        return

    print(f"🚀 Starting Kafka Producer for topic: {TOPIC}")
    p = Producer({'bootstrap.servers': BOOTSTRAP_SERVERS})
    
    test_df = pd.read_csv(DATA_FILE)
    print(f"📊 Loaded {len(test_df)} rows for streaming.")

    try:
        while True: # Infinite loop for simulation
            for _, row in test_df.iterrows():
                payload = {
                    "run_name": row['Run_Name'],
                    "metrics": row.to_dict(),
                    "timestamp": time.time()
                }
                
                p.produce(
                    TOPIC, 
                    value=json.dumps(payload).encode('utf-8'),
                    callback=delivery_report
                )
                p.poll(0)
                
                # Stream speed control
                time.sleep(0.5)
            
            print("🔄 Reached end of data. Restarting stream...")
            
    except KeyboardInterrupt:
        print("🛑 Producer stopped by user.")
    finally:
        p.flush()

if __name__ == "__main__":
    run_producer()
