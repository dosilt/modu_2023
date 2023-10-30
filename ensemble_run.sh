CUDA_VISIBLE_DEVICES=0 python ensemble.py --score_models EleutherAI/polyglot-ko-1.3b EleutherAI/polyglot-ko-5.8b \
 --exp_paths inference/exp1/exp1_inference.jsonl inference/exp2/exp2_inference.jsonl inference/exp3/exp3_inference.jsonl inference/exp4/exp4_inference.jsonl \
 --test_path NIKL_SC_2023/nikluge-sc-2023-test.jsonl --save_name ensemble_v1.jsonl