import json
import os

os.makedirs('prompts', exist_ok=True)

with open('master_prompts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

num_collectors = 10 #Can change thid number to the amount of containers you're using

def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

# Dynamically read every category in your master JSON
categories = {}
for key, value in data.items():
    categories[key] = {
        "repeat": value.get("repeat", 1),
        "chunks": chunk_list(value.get("sessions", []), num_collectors)
    }

# THE FIX: Force 'negative_code' to exist even if missing from the master file
if 'negative_code' not in categories:
    categories['negative_code'] = {
        "repeat": 1,
        "chunks": [[] for _ in range(num_collectors)]
    }

for i in range(num_collectors):
    collector_id = i + 1
    chunk_data = {}
    
    for key, info in categories.items():
        chunk_data[key] = {
            "repeat": info["repeat"],
            "sessions": info["chunks"][i]
        }

    with open(f"prompts/collector-{collector_id}.json", 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, indent=2)

print("Successfully regenerated 10 prompt files with the negative_code key included!")
