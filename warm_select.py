import json
import heapq
import pandas as pd
import numpy as np
import random
import copy
import math


def select(raw_path, FBNM_path, output, num):
    raw = []
    with open(raw_path, 'r') as f:
        for line in f.readlines():
            raw.append(json.loads(line))
    
    new = []
    with open(FBNM_path, 'r') as f:
        for line in f.readlines():
            new.append(json.loads(line))
    
    final = []
    for i, (item_r, item_n) in enumerate(zip(raw, new)):
        assert item_r['conversations'][0]['value'] == item_n['conversations'][0]['value']
        score = item_r['fbnm']*(1-(item_r['fbnm']-item_n['fbnm'])/item_r['fbnm'])
        tmp = copy.deepcopy(item_n)
        tmp['score'] = score
        del tmp['fbnm']
        final.append(tmp)
    
    final.sort(key=lambda x: x['score'], reverse=True)

    with open(output, 'w') as f:
        json.dump(final[:num], f, ensure_ascii=False, indent=4)


# raw_path = 'FBNM/FBNM_llama_response.jsonl'
# fbnm_path = 'FBNM/FBNM_llama_response_warmup.jsonl'
# output = 'FBNM/select/llama_response_selected.json'

# select(raw_path, fbnm_path, output, 5200)

# raw_path = 'FBNM/FBNM_qwen_response.jsonl'
# fbnm_path = 'FBNM/FBNM_qwen_response_warmup.jsonl'
# output = 'FBNM/select/qwen_response_selected.json'

# select(raw_path, fbnm_path, output, 5200)

raw_path = 'FBNM/FBNM_llama2_response.jsonl'
fbnm_path = 'FBNM/FBNM_llama2_response_warmup.jsonl'
output = 'FBNM/select/llama2_response_selected.json'

select(raw_path, fbnm_path, output, 5200)     


    