(
model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/Qwen2-7B-warmup-sbert"
CUDA_VISIBLE_DEVICES=0 python entropy_sentence.py \
        --start 0 --end 60000 \
        --base_model $model \
        --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
        --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/entropy/entropy_qwen_sentence_warmup.jsonl \
        --type chatml
)&
(
model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/Qwen2-7B-warmup-sbert"
CUDA_VISIBLE_DEVICES=1 python entropy_response.py \
        --start 0 --end 60000 \
        --base_model $model \
        --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
        --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/entropy/entropy_qwen_response_warmup.jsonl \
        --type chatml
)&
(
model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/llama3-8B-warmup-sbert"
CUDA_VISIBLE_DEVICES=2 python entropy_response.py \
        --start 0 --end 60000 \
        --base_model $model \
        --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
        --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/entropy/entropy_llama_response_warmup.jsonl \
        
)&
(
model="/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/model/llama3-8B-warmup-sbert"
CUDA_VISIBLE_DEVICES=3 python entropy_sentence.py \
        --start 0 --end 60000 \
        --base_model $model \
        --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json" \
        --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/entropy/entropy_llama_sentence_warmup.jsonl \

)

