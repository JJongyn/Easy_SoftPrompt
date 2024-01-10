

## ddp
for data in 'copa' 'rte' 'wic'; do
for method in dpt residual ; do
for num in 1 2 3; do

model=t5-base

CUDA_VISIBLE_DEVICES=0 python main.py --datasets=${data} --model_name=${model} --enc_prompt_tokens 100 -ts 16 -e 100 --bottle_neck 10 --save_name ${model}_${data}_${method} --method ${method}

done
done
done
