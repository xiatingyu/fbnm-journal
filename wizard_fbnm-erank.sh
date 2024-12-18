source activate dpo



model="/cpfs01/shared/Group-m6/xiatingyu.xty/model/Llama-2-13b-hf"

CUDA_VISIBLE_DEVICES=0,1 python FBNM-response.py \
        --start 0 --end 145000 \
        --base_model $model \
        --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/WizardLM_evol_instruct_V2_143k.json" \
        --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_llama2_response.jsonl 

   
# (
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/model/Qwen2-7B"
# CUDA_VISIBLE_DEVICES=1 python FBNM-response.py \
#         --start 0 --end 145000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/WizardLM_evol_instruct_V2_143k.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_qwen2_sentence.jsonl 

# CUDA_VISIBLE_DEVICES=1 python FBNM-response-erank.py \
#         --start 0 --end 145000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/WizardLM_evol_instruct_V2_143k.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_qwen2_sentence_erank.jsonl 
# )
# &(
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/model/Llama-2-13b-hf"
# CUDA_VISIBLE_DEVICES=2,3 python FBNM-response-erank.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/WizardLM_evol_instruct_V2_143k.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_llama2_response_erank.jsonl \
        
# )&
# (
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/llama2-13B-warmup-sbert"
# CUDA_VISIBLE_DEVICES=1 python FBNM-response.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/WizardLM_evol_instruct_V2_143k.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_llama2_response_warmup.jsonl \
        
# )&
# (
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/llama2-13B-warmup-sbert"
# CUDA_VISIBLE_DEVICES=7 python FBNM-sentence.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/WizardLM_evol_instruct_V2_143k.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_llama2_sentence_warmup.jsonl \

# )

