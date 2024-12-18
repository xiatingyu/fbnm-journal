source activate dpo

# (
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/Qwen2-7B-warmup-sbert"
# CUDA_VISIBLE_DEVICES=0 python FBNM-sentence.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/FBNM_qwen_sentence_warmup.jsonl \
#         --type chatml 
# )&
# (
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/Qwen2-7B-warmup-sbert"
# CUDA_VISIBLE_DEVICES=1 python FBNM-response.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/FBNM_qwen_response_warmup.jsonl \
#         --type chatml 
# )&
# (
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/llama3-8B-warmup-sbert"
# CUDA_VISIBLE_DEVICES=2 python FBNM-response.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/FBNM_llama_response_warmup.jsonl \
        
# )&
# (
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/llama3-8B-warmup-sbert"
# CUDA_VISIBLE_DEVICES=3 python FBNM-sentence.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/FBNM_llama_sentence_warmup.jsonl \

# )&
(
model="/cpfs01/shared/Group-m6/xiatingyu.xty/model/8b-base"
CUDA_VISIBLE_DEVICES=0 python FBNM-response.py \
        --start 0 --end 60000 \
        --base_model $model \
        --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
        --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/FBNM_llama3_response.jsonl \
        
)&
(
model="/cpfs01/shared/Group-m6/xiatingyu.xty/model/Qwen2-7B"
CUDA_VISIBLE_DEVICES=1 python FBNM-response.py \
        --start 0 --end 60000 \
        --base_model $model \
        --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
        --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/FBNM_qwen2_sentence.jsonl \

)
# &(
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/model/Llama-2-13b-hf"
# CUDA_VISIBLE_DEVICES=2,3 python FBNM-response-erank.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/FBNM_llama2_response_erank.jsonl \
        
# )&
# (
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/llama2-13B-warmup-sbert"
# CUDA_VISIBLE_DEVICES=1 python FBNM-response.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/FBNM_llama2_response_warmup.jsonl \
        
# )&
# (
# model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/llama2-13B-warmup-sbert"
# CUDA_VISIBLE_DEVICES=7 python FBNM-sentence.py \
#         --start 0 --end 60000 \
#         --base_model $model \
#         --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
#         --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/FBNM/FBNM_llama2_sentence_warmup.jsonl \

# )

