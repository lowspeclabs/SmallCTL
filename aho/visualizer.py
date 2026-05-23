import json
import time
import os
import argparse
from pathlib import Path

def print_ui(events):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 80)
    print(" AHO Live Runner Visualizer ".center(80, "="))
    print("=" * 80)
    
    recent_events = events[-20:]
    for event in recent_events:
        mod = event.get("module", "unknown")
        act = event.get("action", "")
        msg = event.get("message", "")
        print(f"[{mod}] {act}: {msg}")
        
    print("=" * 80)
    print("Press Ctrl+C to exit")

def stream_logs(log_dir: str):
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory {log_dir} does not exist. Waiting...")
        while not log_path.exists():
            time.sleep(1)
            
    events = []
    
    # Find latest log file
    log_files = sorted(log_path.glob("*.jsonl"))
    if not log_files:
        print("No jsonl logs found in directory.")
        return
        
    target_log = log_files[-1]
    
    with open(target_log, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            try:
                event = json.loads(line)
                events.append(event)
                print_ui(events)
            except json.JSONDecodeError:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="aho/logs/trials")
    args = parser.parse_args()
    
    try:
        stream_logs(args.log_dir)
    except KeyboardInterrupt:
        print("\nExiting AHO Visualizer...")
