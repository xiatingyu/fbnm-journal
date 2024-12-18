import sys
import os
import torch
import transformers
import json
import jsonlines
import argparse
import copy
import numpy as np
import heapq
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
MAX_INT = sys.maxsize

import torch.nn.functional as F
import torch.nn as nn


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default='model/Qwen1.5-1.8B')  # model path
    parser.add_argument("--data_file", type=str, default='/cpfs01/shared/Group-m6/dangkai.dk/workspace/scripts/code_evol_pack_final.jsonl')  # data path
    parser.add_argument("--output_file", type=str, default='tmp.jsonl')  # output path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=10)  # end index
    parser.add_argument("--type", type=str, default="base")  # end index
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("Processing data ...")
    instructions, systems, outputs = [], [], []

    if os.path.exists(args.output_file):
        print("File already exists. ", args.output_file)
    else:
        data = []
        with open(args.data_file,"r", encoding="utf8") as f:
            data = json.load(f)
        
        data = data[args.start:args.end]

        PROMPT_RAW = {
            "system": (
                "SYSTEM: {system}"
            ),
            "user": (
                "\nHuman: {query}\nAssistant: "    
            ),
            "ass": (
                "{response}"
            )
        }
        PROMPT_CHATML = {
            "system": (
                "<|im_start|>system\n{system}<|im_end|>"
            ),
            "user": (
                "\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n" 
            ),
            "ass": (
                "{response}<|im_end|>"
            )
        }

        data_list = []
        for item in data:
            encoded_messages = ""
            for i in range(len(item['conversations'])):
                message = item['conversations'][i]
                if args.type == "chatml":
                    if message["from"] == "human":
                        encoded_messages += PROMPT_CHATML['user'].format(query=message['value'])
                    else:
                        encoded_messages += PROMPT_CHATML['ass'].format(response=message['value'])
                else:
                    if message["from"] == "human":
                        encoded_messages += PROMPT_RAW['user'].format(query=message['value'])
                    else:
                        encoded_messages += PROMPT_RAW['ass'].format(response=message['value'])
            encoded_messages = encoded_messages.strip()
            # response = PROMPT_RAW['ass'].format(response=item['conversations'][-1]['value'])
            data_list.append(encoded_messages)

        
        print("Loading model ...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", output_hidden_states=True).eval()
        
    
        mean_entropies_all = []
        with open(args.output_file, "w+", encoding="utf8") as f:
            for i in tqdm(range(len(data_list))):
                messages = data_list[i]
                input_ids = tokenizer(messages, truncation=True, max_length=4096, return_tensors="pt").to(device)
                input_ids = input_ids['input_ids'].long()
                token_ids = torch.squeeze(input_ids.clone())
    
                with torch.no_grad():
                    output = model(input_ids, labels=token_ids)
                    _, logits = output.loss, output.logits 

                    loss = torch.nn.functional.cross_entropy(torch.squeeze(logits, dim=0), token_ids, ignore_index=-100, reduction='mean')
                    loss = loss.item()
                    mean_entropy = loss
                    mean_entropies_all.append(loss)

                    # probs = F.softmax(logits, dim=1)  # softmax -> logits->probs (seq_len, vocab_size)
                    # entropies = -(torch.log(probs)*probs).sum(dim=1)  # entropy at every token position ->>> (seq_len)
                    
                    # mean_entropy = torch.mean(entropies).item() # avg(entropy)
                    # mean_entropies_all.append(mean_entropy)

                tmp = copy.deepcopy(data[i])
                tmp['id'] = i+int(args.start)
                tmp['sentence_entropy'] = mean_entropy
                json.dump(tmp, f, ensure_ascii=False)
                f.write("\n")
            
                
                
        # assert len(data) == len(data_list)
        
        # final = []
        # for i in range(len(data)):
        #     assert data[i]['conversations'][0]['value'] in data_list[i]
        #     tmp = copy.deepcopy(data[i])
        #     tmp['id'] = i+int(args.start)
        #     tmp['ppl'] = ppl[i]
        #     final.append(tmp)

        # with open(args.output_file, "w+", encoding="utf8") as f:
        #     json.dump(final, f, ensure_ascii=False, indent=4)
            
    