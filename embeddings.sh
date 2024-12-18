# pip install sentence_transformers
export HF_ENDPOINT=https://hf-mirror.com

python sentence_bert_embeddings.py \
    --data_path '/cpfs01/shared/Group-m6/xiatingyu.xty/ZIP/WizardLM_evol_instruct_V2_143k.json' \
    --embedding_cache_path '/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/embeddings/wizardlm_embeddings.npy' 