CUDA_VISIBLE_DEVICES=0 python inference.py --prompt_type 2 --batch_size 1 \
 --save_name exp1_inference.jsonl --output_path inference/exp1 --lora_path results/exp1/lora_weight

CUDA_VISIBLE_DEVICES=0 python inference.py --prompt_type 4 --batch_size 1 \
 --save_name exp2_inference.jsonl --output_path inference/exp2 --lora_path results/exp2/lora_weight

CUDA_VISIBLE_DEVICES=0 python inference.py --prompt_type 4 --batch_size 1 \
 --save_name exp3_inference.jsonl --output_path inference/exp3 --lora_path results/exp3/lora_weight

CUDA_VISIBLE_DEVICES=0 python inference.py --prompt_type 4 --batch_size 1 \
 --save_name exp4_inference.jsonl --output_path inference/exp4 --lora_path results/exp4/lora_weight
