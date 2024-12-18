from apricot import FeatureBasedSelection, FacilityLocationSelection, GraphCutSelection
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
import json

def external_retrival(embeddings, sample_number):
    selector = FacilityLocationSelection(sample_number, metric='cosine', optimizer='naive', verbose=False)
    selector.fit(embeddings)

    data_index = selector.ranking[:sample_number]
    
    return data_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--data_path", type=str, default='/cpfs01/shared/Group-m6/xiatingyu.xty/entropy/cross_entropy/openhermes/openhermes_cross_entropy_llama3_10w.jsonl')
    parser.add_argument("--EMBEDDING_PATH", type=str, default='/cpfs01/shared/Group-m6/xiatingyu.xty/entropy/all_data_embeddings/openhermes/embeddings/openhermes_emb_llama3_10w.npy')
    parser.add_argument("--output_path", type=str, default='/cpfs01/shared/Group-m6/xiatingyu.xty/entropy/sub/openhermes/openhermes_sub_llama3_')
    args = parser.parse_args()
    print('------------------------------------------Submodular selection---------------------------------------------')
    print(vars(args))

    embeddings = np.load(f'{args.EMBEDDING_PATH}')
    print('load embedding ok....', len(embeddings))

    data = []
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    print(len(data))
    number = len(embeddings)
    sub_number = int(number*args.ratio)
    # print('number:', number)

    print('sub_number:', sub_number)
    submodular_id = []
    selected_id = []

    selected_id = external_retrival(embeddings, sub_number)
            
    

    # while len(selected_id) < number:
    #     if selected_id != []:
    #         remain_id = [i for i in range(number) if i not in selected_id]
    #         print('remain_id:', len(remain_id))
    #         sub_embeddings = embeddings[remain_id]
    #         print('select embedding ok....', len(sub_embeddings))
    #         tmp_id = external_retrival(sub_embeddings, sub_number)
            
    #         tmp_selected_id = [remain_id[i] for i in list(tmp_id)]
    #         submodular_id.append(tmp_selected_id)
    #         selected_id.extend(tmp_selected_id)
    #     else:
    #         print('no selected id....')
    #         tmp_id = external_retrival(embeddings, sub_number)
    #         submodular_id.append(tmp_id)
    #         selected_id.extend(tmp_id)
    
    # # remain_id = [i for i in range(number) if i not in selected_id]
    # # submodular_id.append(remain_id)
    # # selected_id.extend(remain_id)
    print(len(set(selected_id)), sub_number)
    assert len(set(selected_id)) == sub_number
    
    # print('submodular_id:', len(submodular_id))
    
    # for i in range(len(submodular_id)):
    #     path = args.output_path + str(i) + '.jsonl'
    with open(args.output_path, 'w') as f:
        for j in selected_id:
            f.write(json.dumps(data[j], ensure_ascii=False) + '\n')
