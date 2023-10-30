from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
from dataload import load_data, CreateDataset, CollateFn
import torch.nn.functional as F
from termcolor import colored

import argparse
import torch
import os


def main(args):
    if os.path.isdir(args.experiment_name):
        print(f'이미 {args.experiment_name}이 존재합니다.')
        return
    
    os.makedirs(args.experiment_name)
    
    ## tokenizer 불러오기 ##
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ## 학습 데이터 불러오기 ##
    train_list, _, _ = load_data(args, 'train')
    train_set = CreateDataset(args, train_list, tokenizer, 'train')
    collate_fn = CollateFn(tokenizer)
    
    ## 모델 불러오기 torch_dtype을 선언 안하면 메모리 초과됨 ##
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    ## Lora는 기본 default setting을 이용함 ##
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    ## 학습 hyper-parameters ##    
    training_args = TrainingArguments(
        output_dir=f'{args.experiment_name}',
        logging_dir=f'./logs/{args.experiment_name}',
        
        ## 학습 중에 validation은 따로 진행 하지 않음.
        # 1. generation은 classification 같은 task에 비해 시간이 너무 오래 걸림
        # 2. generation에 있는 hyper-parameter에 따른 성능 차이가 발생함 (신뢰성 하락)
        ## 따라서 학습 중에 얻은 validation은 시간만 소요되고, 신뢰성이 부족하다고 판단되어 하지 않음.
        do_eval=False,
        
        save_strategy="no",
        # save_total_limit=1,

        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        weight_decay=args.weight_decay,
        bf16=True,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )
    
    trainer.train()
    model.save_pretrained(f'{args.experiment_name}/lora_weight')
    
if __name__ == '__main__':
    argument = argparse.ArgumentParser()
    argument.add_argument('--train_path', default='NIKL_SC_2023/nikluge-sc-2023-train.jsonl', help='train data 경로')
    argument.add_argument('--valid_path', default='NIKL_SC_2023/nikluge-sc-2023-dev.jsonl', help='dev data 경로')
    argument.add_argument('--test_path', default='NIKL_SC_2023/nikluge-sc-2023-test.jsonl', help='test data 경로')
    argument.add_argument('--model_name', default='beomi/llama-2-ko-7b', help='pretrained backbone model') # paust/pko-t5-large, gogamza/kobart-base-v2 beomi/llama-2-ko-7b

    argument.add_argument('--add_valid', action='store_true', help='valid data를 학습에 넣을 것인지 아닌지')
    argument.add_argument('--batch_size', default=2, type=int)
    argument.add_argument('--epochs', default=2, type=int)
    argument.add_argument('--scheduler', default='cosine')
    argument.add_argument('--weight_decay', default=0.01, type=float)
    argument.add_argument('--warmup_ratio', default=0.05, type=float)
    argument.add_argument('--accumulation', default=4, type=int)
    argument.add_argument('--prompt_type', default=0, type=int)
    argument.add_argument('--learning_rate', default=2.5e-4, type=float)
    argument.add_argument('--experiment_name', default='result/exp1', help='저장할 path = result/{experiment_name}')
    
    args = argument.parse_args()
    
    args.train_path = f'{os.path.dirname(os.path.realpath(__file__))}/{args.train_path}'
    args.valid_path = f'{os.path.dirname(os.path.realpath(__file__))}/{args.valid_path}'
    args.test_path = f'{os.path.dirname(os.path.realpath(__file__))}/{args.test_path}'
    
    print(colored('------------------------------------------------------------', 'blue'))
    for key, value in vars(args).items():
        print(colored(f'{key:15}', 'green'), end=' : ')
        print(value)
    print(colored('------------------------------------------------------------', 'blue'))
    
    main(args)
