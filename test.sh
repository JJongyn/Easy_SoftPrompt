

## ddp
for data in 'rte'; do
for num in 1; do

model=t5-small
method=test

CUDA_VISIBLE_DEVICES=1 python main.py --datasets=${data} --model_name=${model} --enc_prompt_tokens 1 -ts 16 -e 100 --bottle_neck 10 --save_name [TEST]_${model}_${data}_${method} --method ${method} --memo test

done
done

