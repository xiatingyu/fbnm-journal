
model="/cpfs01/shared/Group-m6/xiatingyu.xty/model/Llama-2-13b-hf"
CUDA_VISIBLE_DEVICES=0,1,2,3 python FBNM-response-erank.py \
        --start 0 --end 145000 \
        --base_model $model \
        --data_file "/cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/data/WizardLM_evol_instruct_V2_143k.json" \
        --output_file /cpfs01/shared/Group-m6/xiatingyu.xty/journel-alpaca/wizard/FBNM_llama2_response_erank.jsonl 
    