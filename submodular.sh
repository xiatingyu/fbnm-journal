

python Facility.py --ratio 0.02 \
    --data_path '/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json' \
    --EMBEDDING_PATH '/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/embeddings/alpaca_gpt4_embeddings.npy' \
    --output_path 'embeddings/qwen.jsonl'

python Facility.py --ratio 0.02 \
    --data_path '/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/alpaca_gpt4.json' \
    --EMBEDDING_PATH '/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/embeddings/alpaca_gpt4_embeddings.npy' \
    --output_path 'embeddings/llama.jsonl'