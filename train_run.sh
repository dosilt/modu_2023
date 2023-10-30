wandb disabled

CUDA_VISIBLE_DEVICES=0 python main.py --prompt_type 2 --batch_size 2 \
 --accumulation 8 --warmup_ratio 0.05 --learning_rate 2.5e-4 --epochs 2 --experiment_name results/exp1

CUDA_VISIBLE_DEVICES=0 python main.py --prompt_type 4 --batch_size 2 \
 --accumulation 8 --warmup_ratio 0.05 --learning_rate 1.5e-4 --epochs 3 --experiment_name results/exp2

CUDA_VISIBLE_DEVICES=0 python main.py --prompt_type 4 --batch_size 2 \
 --accumulation 8 --warmup_ratio 0.05 --learning_rate 1.0e-4 --epochs 3 --add_valid --experiment_name results/exp3

CUDA_VISIBLE_DEVICES=0 python main.py --prompt_type 4 --batch_size 2 \
 --accumulation 8 --warmup_ratio 0.05 --learning_rate 2.0e-4 --epochs 3 --add_valid --experiment_name results/exp4
