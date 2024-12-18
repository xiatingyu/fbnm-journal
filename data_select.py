import json
import heapq
import pandas as pd
import numpy as np
import random
import copy
import math


def kmeans_select(path, FBNM_path, output, num):
    data = []
    with open(path, 'r') as f:
        data = json.load(f)
        # for line in f.readlines():
        #     data.append(json.loads(line))

    FBNM = []
    with open(FBNM_path, 'r') as f:
        for line in f.readlines():
            FBNM.append(json.loads(line))
        # FBNM = json.load(f)
    print(len(FBNM))


    print(len(data))
    kmeans_dict = {}
    kmeans_num = {}
    for item in data:
        id = item['id']
        item['fbnm'] = FBNM[id]['fbnm']
        cluster_center = item['cluster_center']
        if item['cluster'] not in kmeans_dict:
            kmeans_dict[item['cluster']] = []
        kmeans_dict[item['cluster']].append(item)
        if item['cluster'] not in kmeans_num:
            kmeans_num[item['cluster']] = 0
        kmeans_num[item['cluster']] += 1

    
    cluster_num = len(kmeans_dict)
    number = num // cluster_num
    all_data = len(data) 

    final = []
    for key in kmeans_dict:
        sub_list = kmeans_dict[key]
        sub_list.sort(key=lambda x: x['fbnm'], reverse=True)
        sub_number = math.ceil(num * (kmeans_num[key] / all_data))
        
        final.extend(sub_list[:sub_number])

        # final.extend(kmeans_dict[key][:number])

    import random
    random.shuffle(final)
    print(len(final))
    with open(output, 'w') as f:
        for item in final[:num]:
            # print(item)
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    return


data = []
with open('/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/select-1204/FBNM_qwen_response.json','r')as f:
    data = json.load(f)

data = sorted(data, key=lambda x: x['fbnm'])
with open('/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/select-1204/FBNM_qwen_response_1.json','w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
# with open('/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_llama3_response_erank.jsonl','r') as f:
#     for item in f:
#         data.append(json.loads(item))
# data = sorted(data, key=lambda x: x['fbnm_erank'])

# with open('/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/select/FBNM_llama3_response_erank.json','w') as f:
#     json.dump(data[:10000], f, ensure_ascii=False, indent=4)

# data = []
# with open('/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_llama2_response_erank.jsonl','r') as f:
#     for item in f:
#         data.append(json.loads(item))
# data = sorted(data, key=lambda x: x['fbnm_erank'])

# with open('/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/select/FBNM_llama2_response_erank.json','w') as f:
#     json.dump(data[:10000], f, ensure_ascii=False, indent=4)

# data = []
# with open('/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_qwen2_response_erank.jsonl','r') as f:
#     for item in f:
#         data.append(json.loads(item))
# data = sorted(data, key=lambda x: x['fbnm_erank'])

# with open('/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/select/FBNM_qwen2_response_erank.json','w') as f:
#     json.dump(data[:10000], f, ensure_ascii=False, indent=4)
# FBNM_path='/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/fbnm/FBNM_qwen_response.jsonl'
# path='/cpfs01/shared/Group-m6/xiatingyu.xty/nuggets/datasets/alpaca_gpt4_qwen/alpaca_gpt4_kmeans_label.json'
# num=10000
# output='fbnm/FBNM_qwen_response_kmeans.jsonl'
# kmeans_select(path, FBNM_path, output, num)

# FBNM_path='/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/fbnm/FBNM_llama_response.jsonl'
# path='/cpfs01/shared/Group-m6/xiatingyu.xty/nuggets/datasets/alpaca_gpt4/alpaca_gpt4_kmeans_label.json'
# num=10000
# output='fbnm/FBNM_llama_response_kmeans.jsonl'
# kmeans_select(path, FBNM_path, output, num)

# data = []
# with open('fbnm/FBNM_llama_sentence.jsonl', 'r') as f:
#     for line in f.readlines():
#         data.append(json.loads(line))

# sentence = []
# with open('fbnm/FBNM_qwen_sentence.jsonl', 'r') as f:
#     for line in f.readlines():
#         sentence.append(json.loads(line))


# data = sorted(data, key=lambda x: x['fbnm'], reverse=True)
# num = int(len(data)*0.1)
# new = data[:num]
# # random.shuffle(new)
# with open('fbnm/FBNM_llama_sentence_0.1.jsonl', 'w') as f:
#     for item in new:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')

# data = sorted(sentence, key=lambda x: x['fbnm'], reverse=True)
# num = int(len(data)*0.1)
# new = data[:num]
# # random.shuffle(new)
# with open('fbnm/FBNM_qwen_sentence_0.1.jsonl', 'w') as f:
#     for item in new:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')
