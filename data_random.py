import random
import json

path = 'fbnm/fbnm_qwen_response_0.1.jsonl'
data = []
with open(path,'r')as f:
    for item in f:
        data.append(json.loads(item))

random.shuffle(data)
with open('FBNM/select/qwen-random-1.json','w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

path = 'fbnm/fbnm_llama_response_0.1.jsonl'
data = []
with open(path,'r')as f:
    for item in f:
        data.append(json.loads(item))

random.shuffle(data)
with open('FBNM/select/llama-random-1.json','w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)